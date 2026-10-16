"""
train.py -- Train, validate, test and save a segmentation model on HipMRI 2D.

The same script trains both models so they are compared under identical
conditions (same data, splits, augmentation, loss, optimiser and schedule):

    python train.py --model baseline --epochs 50
    python train.py --model improved --epochs 50

Model selection uses the VALIDATION prostate Dice only. The test split is
evaluated exactly once, at the end, with the selected checkpoint -- so the
reported test score is not biased by any training decision.

Outputs (in --out, default runs/<model>/):
    best.pt            weights of the best epoch (git-ignored, never committed)
    history.json       per-epoch losses, per-class validation Dice, timings
    curves.png         loss + per-class validation Dice plot
    results.json       final test metrics + resource profile (params, VRAM, latency)
"""
import argparse
import os
import time

import torch
from torch.amp import GradScaler, autocast

from dataset import DEFAULT_ROOT, get_dataloaders
from modules import DiceCELoss, build_model, count_parameters
from utils import SegMetrics, measure_latency, plot_history, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a HipMRI prostate segmentation model")
    p.add_argument("--model", choices=["baseline", "improved"], required=True)
    p.add_argument("--root", default=DEFAULT_ROOT, help="keras_slices_data directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="use only N slices per split (smoke tests)")
    p.add_argument("--out", default=None, help="output directory (default runs/<model>)")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    metrics = SegMetrics()
    total_loss, n_batches = 0.0, 0
    use_amp = device.type == "cuda"

    for images, masks, _ in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits.float(), masks)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics.update(logits.float(), masks)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1), metrics.summary()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.out or os.path.join("runs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    t_load = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(
        args.root, batch_size=args.batch_size, num_workers=args.workers, limit=args.limit)
    print(f"Data loaded in {time.time() - t_load:.0f}s | train {len(train_loader.dataset)} | "
          f"val {len(val_loader.dataset)} | test {len(test_loader.dataset)} slices")

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | {n_params:,} trainable parameters")

    criterion = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "epoch_time_s": []}
    best_dice, best_epoch = -1.0, 0
    ckpt_path = os.path.join(out_dir, "best.pt")
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val["dice"])
        history["epoch_time_s"].append(epoch_time)

        improved = val["prostate_dice"] > best_dice
        if improved:
            best_dice, best_epoch = val["prostate_dice"], epoch
            torch.save(model.state_dict(), ckpt_path)
        d = val["dice"]
        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"prostate {d['prostate']:.4f} | bladder {d['bladder']:.3f} | rectum {d['rectum']:.3f} | "
              f"bone {d['bone']:.3f} | {epoch_time:.0f}s" + ("  *best*" if improved else ""), flush=True)
        save_json(history, os.path.join(out_dir, "history.json"))
        plot_history(history, os.path.join(out_dir, "curves.png"), f"{args.model} U-Net on HipMRI")

    train_time = time.time() - t_train
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    # ---- Final evaluation: best checkpoint, held-out test split, exactly once ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test = run_epoch(model, test_loader, criterion, device)
    latency_ms = measure_latency(model, device)

    results = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_prostate_dice": best_dice,
        "test": test,
        "test_loss": test_loss,
        "resources": {
            "parameters": n_params,
            "peak_train_vram_gb": peak_vram_gb,
            "mean_epoch_time_s": sum(history["epoch_time_s"]) / len(history["epoch_time_s"]),
            "total_train_time_s": train_time,
            "inference_ms_per_slice": latency_ms,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "config": vars(args),
    }
    save_json(results, os.path.join(out_dir, "results.json"))

    print("\n" + "=" * 70)
    print(f"{args.model.upper()} -- best epoch {best_epoch} (val prostate Dice {best_dice:.4f})")
    print("HELD-OUT TEST SET (evaluated once):")
    for name in test["dice"]:
        print(f"   {name:10s}  Dice {test['dice'][name]:.4f}   IoU {test['iou'][name]:.4f}")
    target = "PASS" if test["prostate_dice"] >= 0.75 else "below 0.75"
    print(f"   prostate Dice target (>= 0.75): {test['prostate_dice']:.4f} [{target}]")
    r = results["resources"]
    vram = f"{r['peak_train_vram_gb']:.2f} GB" if r["peak_train_vram_gb"] is not None else "n/a (cpu)"
    print(f"RESOURCES: {r['parameters']:,} params | peak VRAM {vram} | "
          f"{r['mean_epoch_time_s']:.1f}s/epoch | {r['inference_ms_per_slice']:.2f} ms/slice inference")
    print("=" * 70)
    print(f"Saved to {out_dir}/: best.pt, history.json, curves.png, results.json")


if __name__ == "__main__":
    main()
