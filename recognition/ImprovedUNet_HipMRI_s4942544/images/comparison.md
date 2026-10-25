Mean ± standard deviation over random seeds (2D U-Net (baseline): 3, 2D Improved U-Net: 3), held-out test split (3 patients, 540 slices).

**Prostate accuracy and clinical error modes**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net |
|---|---|---|
| Prostate Dice (all slices) | 0.873 ± 0.003 | 0.871 ± 0.007 |
| Prostate IoU | 0.775 ± 0.004 | 0.771 ± 0.010 |
| Prostate Dice: central slices | 0.906 ± 0.001 | 0.909 ± 0.003 |
| Prostate Dice: edge slices | 0.741 ± 0.013 | 0.713 ± 0.023 |
| Prostate boundary F-score (2 px) | 0.920 ± 0.005 | 0.915 ± 0.010 |
| Under-contouring (% of prostate missed) | 12.2 ± 2.1 | 12.7 ± 0.6 |
| Leak into bladder (% of prostate area) | 0.92 ± 0.31 | 1.17 ± 0.25 |
| Leak into rectum (% of prostate area) | 0.26 ± 0.28 | 0.16 ± 0.03 |
| False-positive prostate-free slices (%) | 0.6 ± 0.6 | 0.6 ± 0.2 |

**Per-class Dice**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net |
|---|---|---|
| body Dice | 0.980 ± 0.000 | 0.981 ± 0.000 |
| bone Dice | 0.938 ± 0.000 | 0.940 ± 0.001 |
| bladder Dice | 0.964 ± 0.002 | 0.964 ± 0.001 |
| rectum Dice | 0.852 ± 0.008 | 0.858 ± 0.016 |
| prostate Dice | 0.873 ± 0.003 | 0.871 ± 0.007 |

**Resource profile**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net |
|---|---|---|
| Parameters (M) | 7.76 ± 0.00 | 3.19 ± 0.00 |
| Peak training VRAM (GB) | 0.89 ± 0.00 | 0.67 ± 0.00 |
| Training time per epoch (s) | 34.5 ± 0.1 | 38.3 ± 0.2 |
| Total training time (min) | 29.2 ± 0.1 | 32.3 ± 0.2 |
| Inference per slice (ms; 3D: volume time / slices) | 1.08 ± 0.12 | 0.93 ± 0.02 |
| Inference per scan (s) | 0.06 ± 0.01 | 0.06 ± 0.00 |
| Peak inference VRAM (GB) | 0.97 ± 0.00 | 0.59 ± 0.00 |
