"""
predict.py -- Load a trained model, run inference on the held-out test split,
answer the research question, and produce the report figures.

    python predict.py --model improved --ckpt runs/improved_s0/best.pt --out eval/improved_s0

Beyond the standard per-class Dice/IoU, the evaluation is STRATIFIED, because a
good global Dice can hide failures where the prostate is small:

  * Edge vs central slices. For every test scan, the prostate's extent is the run
    of slices from its first to its last ground-truth appearance. The outer 20% of
    that extent at each end are "edge" slices (where the gland is small and its
    boundary is ambiguous); the rest are "central".
  * Dice by position within the prostate extent (10 bins), for the position plot.
  * Boundary F-score: agreement of the prostate CONTOUR within a 2-pixel tolerance,
    which penalises ragged or shifted edges that area-overlap Dice barely notices.
  * Clinical error modes: under-contouring (missed prostate), leakage into the
    bladder and rectum (organs at risk), and false-positive prostate on slices
    that contain none.

All thresholds (EDGE_FRACTION, FP_MIN_PIXELS, BOUNDARY_TOL) are fixed here in
advance and were not tuned on the test set.

Outputs (in --out):
    eval.json            every metric above, per-patient Dice, resource profile
    examples.png         input / ground truth / prediction for typical slices
    failure_cases.png    3-5 worst cases with false-positive / false-negative overlays
    dice_by_position.png prostate Dice across the prostate's extent
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless plotting on Rangpur
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset import CLASS_NAMES, DEFAULT_ROOT, NUM_CLASSES, PROSTATE, HipMRISliceDataset
from modules import build_model, count_parameters

EDGE_FRACTION = 0.2  # outer 20% of each scan's prostate extent counts as "edge"
FP_MIN_PIXELS = 20   # a prediction smaller than this on a prostate-free slice is ignored as noise
BOUNDARY_TOL = 2     # boundary F-score tolerance, in pixels
N_POSITION_BINS = 10
BLADDER, RECTUM = 3, 4

# Colours for the six labels in figures: background transparent, then body..prostate.
LABEL_CMAP = ListedColormap(["#000000", "#d9a066", "#4caf50", "#e57373", "#7986cb", "#ffeb3b"])


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained HipMRI segmentation model")
    p.add_argument("--model", choices=["baseline", "improved"], required=True)
    p.add_argument("--ckpt", required=True, help="path to best.pt written by train.py")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="evaluate only N slices (debugging)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model, ds, device, batch_size):
    """
    Predict every test slice, one scan (patient + week) at a time, so the
    per-case inference time the specification asks for can be measured directly.
    Returns predicted label maps (N, H, W) uint8 and a list of per-scan timings.
    """
    model.eval()
    scans = defaultdict(list)
    for i, m in enumerate(ds.meta):
        scans[(m["patient"], m["week"])].append(i)

    preds = np.zeros(ds.masks.shape, dtype=np.uint8)
    timings = []
    for scan, idx in sorted(scans.items()):
        x = torch.from_numpy(ds.images[idx].astype(np.float32)).unsqueeze(1).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = [model(x[i:i + batch_size]).argmax(1) for i in range(0, len(idx), batch_size)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append({"scan": "/".join(scan), "slices": len(idx), "seconds": time.perf_counter() - t0})
        preds[idx] = torch.cat(out).cpu().numpy().astype(np.uint8)
    return preds, timings


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def overlap(pred, true):
    """Dataset-level Dice and IoU for two boolean arrays (every pixel counted once)."""
    inter = np.logical_and(pred, true).sum()
    p, t = pred.sum(), true.sum()
    dice = 2 * inter / (p + t) if (p + t) > 0 else float("nan")
    iou = inter / (p + t - inter) if (p + t - inter) > 0 else float("nan")
    return float(dice), float(iou)


def boundary_fscore(pred, true, tol=BOUNDARY_TOL):
    """
    Contour agreement for binary masks (N, H, W). A boundary pixel counts as
    matched if a boundary pixel of the other mask lies within `tol` pixels.
    Erosion/dilation are done with max-pooling, so no extra libraries are needed.
    """
    def boundary(m):
        m = torch.from_numpy(m).float().unsqueeze(1)
        eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
        return (m - eroded).clamp(min=0)

    def dilate(b):
        return F.max_pool2d(b, 2 * tol + 1, stride=1, padding=tol)

    bp, bt = boundary(pred), boundary(true)
    precision_hits = (bp * dilate(bt)).sum().item()
    recall_hits = (bt * dilate(bp)).sum().item()
    n_pred, n_true = bp.sum().item(), bt.sum().item()
    if n_true == 0 and n_pred == 0:  # nothing to find and nothing predicted: undefined
        return {"precision": float("nan"), "recall": float("nan"), "f_score": float("nan")}
    precision = precision_hits / n_pred if n_pred else 0.0  # predicting nothing earns no credit
    recall = recall_hits / n_true if n_true else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f_score": f}


def prostate_strata(ds):
    """
    Label each slice 'edge', 'central' or 'outside' relative to its scan's
    ground-truth prostate extent, and give its normalised position (0-1) in that extent.
    """
    strata = np.array(["outside"] * len(ds), dtype=object)
    position = np.full(len(ds), np.nan)
    has_prostate = (ds.masks == PROSTATE).any(axis=(1, 2))
    scans = defaultdict(list)
    for i, m in enumerate(ds.meta):
        scans[(m["patient"], m["week"])].append(i)

    for idx in scans.values():
        idx = sorted(idx, key=lambda i: ds.meta[i]["slice"])
        present = [ds.meta[i]["slice"] for i in idx if has_prostate[i]]
        if not present:
            continue
        lo, hi = min(present), max(present)
        extent = hi - lo + 1
        k = max(1, int(round(EDGE_FRACTION * extent)))
        for i in idx:
            s = ds.meta[i]["slice"]
            if lo <= s <= hi:
                strata[i] = "edge" if (s < lo + k or s > hi - k) else "central"
                position[i] = (s - lo) / max(extent - 1, 1)
    return strata, position, has_prostate


def evaluate(ds, preds):
    """Compute every metric reported in the README."""
    masks = ds.masks
    results = {"per_class": {}}
    for c, name in enumerate(CLASS_NAMES):
        d, i = overlap(preds == c, masks == c)
        results["per_class"][name] = {"dice": d, "iou": i}

    strata, position, has_prostate = prostate_strata(ds)
    pp, tp = preds == PROSTATE, masks == PROSTATE

    results["prostate_by_stratum"] = {}
    for name in ["central", "edge"]:
        sel = strata == name
        d, i = overlap(pp[sel], tp[sel])
        results["prostate_by_stratum"][name] = {"dice": d, "iou": i, "slices": int(sel.sum())}

    bins = np.linspace(0, 1, N_POSITION_BINS + 1)
    by_pos = []
    for b in range(N_POSITION_BINS):
        sel = (position >= bins[b]) & ((position < bins[b + 1]) | (b == N_POSITION_BINS - 1 and position == 1))
        by_pos.append(overlap(pp[sel], tp[sel])[0] if sel.any() else float("nan"))
    results["prostate_dice_by_position"] = {"bin_centres": ((bins[:-1] + bins[1:]) / 2).tolist(), "dice": by_pos}

    results["prostate_boundary"] = boundary_fscore(pp, tp)

    gt_area = tp.sum()
    no_prostate = ~has_prostate
    fp_slices = (pp[no_prostate].sum(axis=(1, 2)) > FP_MIN_PIXELS)
    results["clinical_errors"] = {
        "under_contouring_pct": float(100 * np.logical_and(tp, ~pp).sum() / gt_area),
        "leak_into_bladder_pct": float(100 * np.logical_and(pp, masks == BLADDER).sum() / gt_area),
        "leak_into_rectum_pct": float(100 * np.logical_and(pp, masks == RECTUM).sum() / gt_area),
        "false_positive_slice_rate_pct": float(100 * fp_slices.mean()) if no_prostate.any() else float("nan"),
        "prostate_free_slices": int(no_prostate.sum()),
    }

    per_patient = {}
    for p in sorted({m["patient"] for m in ds.meta}):
        sel = np.array([m["patient"] == p for m in ds.meta])
        per_patient[p] = overlap(pp[sel], tp[sel])[0]
    results["prostate_dice_per_patient"] = per_patient

    per_slice = []
    for i in range(len(ds)):
        per_slice.append({
            "index": i, "stratum": strata[i],
            "gt_px": int(tp[i].sum()), "pred_px": int(pp[i].sum()),
            "dice": overlap(pp[i], tp[i])[0],
        })
    return results, per_slice


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def show(ax, img, title=None):
    ax.imshow(img.T, cmap="gray", origin="lower")  # .T + origin='lower' gives the anatomical view
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def overlay_labels(ax, labels):
    ax.imshow(np.ma.masked_where(labels.T == 0, labels.T), cmap=LABEL_CMAP, vmin=0, vmax=5,
              alpha=0.55, origin="lower", interpolation="nearest")


def overlay_errors(ax, pred, true):
    """Prostate errors: false negatives (missed) blue, false positives (extra) red, agreement green."""
    rgba = np.zeros((*pred.T.shape, 4))
    p, t = (pred == PROSTATE).T, (true == PROSTATE).T
    rgba[p & t] = [0.2, 0.8, 0.2, 0.7]
    rgba[t & ~p] = [0.1, 0.4, 1.0, 0.9]
    rgba[p & ~t] = [1.0, 0.15, 0.15, 0.9]
    ax.imshow(rgba, origin="lower", interpolation="nearest")


def plot_examples(ds, preds, per_slice, path, n=3):
    central = [s for s in per_slice if s["stratum"] == "central" and s["gt_px"] > 0]
    central.sort(key=lambda s: s["dice"])
    picks = [central[int(q * (len(central) - 1))] for q in np.linspace(0.3, 0.9, n)] if central else per_slice[:n]
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(picks):
        i = s["index"]
        m = ds.meta[i]
        show(axes[r, 0], ds.images[i].astype(np.float32), f"case {m['patient']} wk {m['week']} slice {m['slice']}")
        show(axes[r, 1], ds.images[i].astype(np.float32), "ground truth")
        overlay_labels(axes[r, 1], ds.masks[i])
        show(axes[r, 2], ds.images[i].astype(np.float32), f"prediction (prostate Dice {s['dice']:.2f})")
        overlay_labels(axes[r, 2], preds[i])
    plt.suptitle("Example test predictions: body, bone, bladder, rectum, prostate (yellow)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def pick_failure_cases(per_slice):
    """3 worst edge slices, the worst central slice, and the worst false-positive slice."""
    edge = sorted([s for s in per_slice if s["stratum"] == "edge" and s["gt_px"] > 0], key=lambda s: s["dice"])
    central = sorted([s for s in per_slice if s["stratum"] == "central" and s["gt_px"] > 0], key=lambda s: s["dice"])
    fp = sorted([s for s in per_slice if s["gt_px"] == 0 and s["pred_px"] > FP_MIN_PIXELS],
                key=lambda s: -s["pred_px"])
    cases = [("edge", s) for s in edge[:3]] + [("central", s) for s in central[:1]] + [("false positive", s) for s in fp[:1]]
    return cases


def plot_failures(ds, preds, cases, path):
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 4, figsize=(12, 2.8 * len(cases)))
    axes = np.atleast_2d(axes)
    described = []
    for r, (kind, s) in enumerate(cases):
        i = s["index"]
        m = ds.meta[i]
        img = ds.images[i].astype(np.float32)
        show(axes[r, 0], img, f"{kind}: case {m['patient']} wk {m['week']} slice {m['slice']}")
        show(axes[r, 1], img, f"ground truth ({s['gt_px']} px prostate)")
        overlay_labels(axes[r, 1], ds.masks[i])
        show(axes[r, 2], img, f"prediction ({s['pred_px']} px prostate)")
        overlay_labels(axes[r, 2], preds[i])
        show(axes[r, 3], img, "prostate errors" + ("" if np.isnan(s["dice"]) else f" (Dice {s['dice']:.2f})"))
        overlay_errors(axes[r, 3], preds[i], ds.masks[i])

        # Zoom the error panel onto the prostate region so small errors are visible.
        region = np.argwhere((preds[i] == PROSTATE) | (ds.masks[i] == PROSTATE))
        if len(region):
            (x0, y0), (x1, y1) = region.min(0), region.max(0)
            pad = 18
            axes[r, 3].set_xlim(max(x0 - pad, 0), min(x1 + pad, img.shape[0]))
            axes[r, 3].set_ylim(max(y0 - pad, 0), min(y1 + pad, img.shape[1]))
        described.append({"kind": kind, "patient": m["patient"], "week": m["week"], "slice": m["slice"],
                          "gt_px": s["gt_px"], "pred_px": s["pred_px"], "dice": s["dice"]})
    plt.suptitle("Failure cases. Error panel (zoomed): green = correct, blue = missed (FN), red = extra (FP)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return described


def plot_position(results, path, title):
    d = results["prostate_dice_by_position"]
    plt.figure(figsize=(7, 4))
    plt.plot(d["bin_centres"], d["dice"], "o-")
    lo, hi = EDGE_FRACTION, 1 - EDGE_FRACTION
    plt.axvspan(0, lo, color="red", alpha=0.08, label="edge slices")
    plt.axvspan(hi, 1, color="red", alpha=0.08)
    plt.axhline(0.75, color="red", ls="--", lw=1, label="0.75 target")
    plt.xlabel("position within the scan's prostate extent (0 = first slice, 1 = last)")
    plt.ylabel("prostate Dice")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    ds = HipMRISliceDataset(args.root, "test", augment=False, limit=args.limit)
    print(f"Loaded {args.model} from {args.ckpt} | {len(ds)} test slices | device {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    preds, timings = run_inference(model, ds, device, args.batch_size)
    results, per_slice = evaluate(ds, preds)

    total_s = sum(t["seconds"] for t in timings)
    results["resources"] = {
        "parameters": count_parameters(model),
        "inference_s_per_scan": total_s / len(timings),
        "inference_ms_per_slice": 1000 * total_s / len(ds),
        "peak_inference_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    train_results = os.path.join(os.path.dirname(args.ckpt), "results.json")
    if os.path.exists(train_results):  # carry over training time / VRAM recorded by train.py
        with open(train_results) as f:
            results["training"] = json.load(f).get("resources", {})
    results["model"] = args.model
    results["test_set"] = {"slices": len(ds), "patients": len({m["patient"] for m in ds.meta}),
                           "scans": len(timings)}
    results["checkpoint"] = args.ckpt
    results["settings"] = {"edge_fraction": EDGE_FRACTION, "fp_min_pixels": FP_MIN_PIXELS,
                           "boundary_tolerance_px": BOUNDARY_TOL}

    plot_examples(ds, preds, per_slice, os.path.join(args.out, "examples.png"))
    results["failure_cases"] = plot_failures(ds, preds, pick_failure_cases(per_slice),
                                             os.path.join(args.out, "failure_cases.png"))
    plot_position(results, os.path.join(args.out, "dice_by_position.png"), f"{args.model} U-Net: prostate Dice by position")
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Printed summary ----
    print("\nPer-class test Dice / IoU:")
    for name, v in results["per_class"].items():
        print(f"   {name:10s}  Dice {v['dice']:.4f}   IoU {v['iou']:.4f}")
    s = results["prostate_by_stratum"]
    print(f"\nProstate Dice  central {s['central']['dice']:.4f} ({s['central']['slices']} slices)"
          f"  |  edge {s['edge']['dice']:.4f} ({s['edge']['slices']} slices)")
    b = results["prostate_boundary"]
    print(f"Prostate boundary F-score (+/-{BOUNDARY_TOL}px): {b['f_score']:.4f}")
    c = results["clinical_errors"]
    print(f"Under-contouring {c['under_contouring_pct']:.1f}% | leak into bladder {c['leak_into_bladder_pct']:.2f}% | "
          f"rectum {c['leak_into_rectum_pct']:.2f}% | false-positive slices {c['false_positive_slice_rate_pct']:.1f}%")
    print("Per-patient prostate Dice:", {k: round(v, 3) for k, v in results["prostate_dice_per_patient"].items()})
    r = results["resources"]
    print(f"Inference: {r['inference_ms_per_slice']:.2f} ms/slice, {r['inference_s_per_scan']:.2f} s/scan")
    print(f"\nSaved to {args.out}/: eval.json, examples.png, failure_cases.png, dice_by_position.png")


if __name__ == "__main__":
    main()
