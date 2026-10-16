"""
dataset.py -- Data loading and preprocessing for HipMRI 2D prostate segmentation.

Design decisions (each one is backed by data_audit.py on the real dataset):

* Splits: the provided keras_slices_{train,validate,test} folders are used as-is.
  The audit confirmed they are already patient-level (31 / 4 / 3 patients, zero
  overlap), so no slice from a test patient is ever seen during training.
* Geometry: 99.5% of slices are 256x128, but one scan is 256x144. Every slice is
  centre-cropped / zero-padded to 256x128. Cropping is used instead of resizing so
  that mask labels are never interpolated and anatomical scale is preserved.
* Intensity: raw MR intensities are unitless and vary between scans (audit: max
  582-738), so each slice is z-score normalised individually.
* Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate. Masks are
  returned as class indices; `to_one_hot` produces the multi-channel encoding used
  by the Dice loss.
* Augmentation (training only): random flips, small rotations/scaling and intensity
  jitter. Geometric transforms are applied identically to image and mask, with
  nearest-neighbour sampling for the mask so no invalid labels are created.

Every item also carries metadata (patient, week, slice index, relative position in
its scan). This is what later allows failures to be analysed at the prostate base
and apex -- the end slices that the research question focuses on.
"""
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
TARGET_SHAPE = (256, 128)
NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5

SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# Images: case_004_week_0_slice_12.nii.gz   Masks: seg_004_week_0_slice_12.nii.gz
NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)")


def _parse_key(filename):
    m = NAME_RE.search(filename)
    return None if m is None else (m.group(1), m.group(2), int(m.group(3)))


def _load_2d(path):
    """Load a Nifti slice as a 2D float array (drop the occasional trailing singleton dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def crop_or_pad(arr, target=TARGET_SHAPE):
    """Centre-crop or zero-pad a 2D array to `target` without any interpolation."""
    out = np.zeros(target, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for dim, size in enumerate(target):
        n = arr.shape[dim]
        if n >= size:  # crop
            start = (n - size) // 2
            src_slices.append(slice(start, start + size))
            dst_slices.append(slice(0, size))
        else:  # pad
            start = (size - n) // 2
            src_slices.append(slice(0, n))
            dst_slices.append(slice(start, start + n))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def normalise(img):
    """Per-slice z-score normalisation (robust to scan-to-scan intensity differences)."""
    return (img - img.mean()) / (img.std() + 1e-8)


def to_one_hot(mask, num_classes=NUM_CLASSES):
    """(N, H, W) class-index mask -> (N, C, H, W) one-hot float tensor."""
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()


class Augment:
    """
    Paired, label-safe augmentation for one (image, mask) sample.

    Image: bilinear resampling. Mask: nearest-neighbour resampling, so a pixel can
    only ever take an existing label value.
    """

    def __init__(self, max_rotate_deg=10.0, scale_range=(0.9, 1.1), p_flip=0.5,
                 intensity_jitter=0.1):
        self.max_rotate = math.radians(max_rotate_deg)
        self.scale_range = scale_range
        self.p_flip = p_flip
        self.intensity_jitter = intensity_jitter

    def __call__(self, image, mask):
        # image: (1, H, W) float, mask: (H, W) long
        if torch.rand(1).item() < self.p_flip:
            # Mirror across the left-right axis. Pelvic anatomy is approximately
            # left-right symmetric, so this produces realistic new examples.
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        angle = (torch.rand(1).item() * 2 - 1) * self.max_rotate
        scale = self.scale_range[0] + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
        cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, *image.shape[-2:]), align_corners=False)

        image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).squeeze(0)
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                             padding_mode="zeros", align_corners=False)[0, 0].long()

        # Mild contrast/brightness jitter mimics coil and scanner variation.
        gain = 1 + (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
        bias = (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
        image = image * gain + bias
        return image, mask


class HipMRISliceDataset(Dataset):
    """
    2D HipMRI slices held in memory.

    Returns (image, mask, meta):
        image: (1, 256, 128) float32, z-score normalised
        mask:  (256, 128) int64 class indices in [0, 5]
        meta:  dict with patient, week, slice, rel_pos (0 = first slice, 1 = last)
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", augment=False, limit=None, num_threads=8):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(SPLIT_DIRS)}")
        img_dir, seg_dir = (os.path.join(root, d) for d in SPLIT_DIRS[split])

        img_files = {_parse_key(f): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
        seg_files = {_parse_key(f): f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")}
        keys = sorted(k for k in img_files if k is not None and k in seg_files)
        if not keys:
            raise RuntimeError(f"No image/mask pairs found in {img_dir} and {seg_dir}")
        if limit:
            keys = keys[:limit]

        # Number of slices in each scan, to express each slice's relative position.
        slices_per_scan = {}
        for p, w, s in keys:
            slices_per_scan[(p, w)] = max(slices_per_scan.get((p, w), 0), s + 1)

        def load_pair(key):
            img = crop_or_pad(_load_2d(os.path.join(img_dir, img_files[key])).astype(np.float32))
            seg = crop_or_pad(_load_2d(os.path.join(seg_dir, seg_files[key])).astype(np.uint8))
            return normalise(img).astype(np.float16), seg

        # Decompressing ~11k .nii.gz files is I/O + zlib bound; threads parallelise it.
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            pairs = list(pool.map(load_pair, keys))

        # float16 storage halves RAM (~0.75 GB for the training split); items are
        # converted back to float32 on access.
        self.images = np.stack([p[0] for p in pairs])
        self.masks = np.stack([p[1] for p in pairs])
        self.meta = [
            {"patient": p, "week": w, "slice": s,
             "rel_pos": s / max(slices_per_scan[(p, w)] - 1, 1)}
            for p, w, s in keys
        ]
        self.augment = Augment() if augment else None
        self.split = split

        bad = set(np.unique(self.masks).tolist()) - set(range(NUM_CLASSES))
        if bad:
            raise ValueError(f"Unexpected label values in {split} masks: {sorted(bad)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return image, mask, self.meta[idx]


def get_dataloaders(root=DEFAULT_ROOT, batch_size=16, num_workers=4, limit=None):
    """Build train (augmented, shuffled), validation and test loaders."""
    train_ds = HipMRISliceDataset(root, "train", augment=True, limit=limit)
    val_ds = HipMRISliceDataset(root, "validate", augment=False, limit=limit)
    test_ds = HipMRISliceDataset(root, "test", augment=False, limit=limit)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


if __name__ == "__main__":
    # Quick self-check: python dataset.py [--root PATH]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    ds = HipMRISliceDataset(args.root, "train", augment=True, limit=args.limit)
    img, mask, meta = ds[0]
    print(f"{len(ds)} slices loaded | image {tuple(img.shape)} {img.dtype} | mask {tuple(mask.shape)} {mask.dtype}")
    print(f"image mean={img.mean():.3f} std={img.std():.3f} | labels present={torch.unique(mask).tolist()}")
    print("meta:", meta)
