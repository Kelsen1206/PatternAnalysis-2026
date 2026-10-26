Mean ± standard deviation over random seeds (2D U-Net (baseline): 3, 2D Improved U-Net: 3, 3D Improved U-Net: 1, 3D Improved U-Net (half resolution): 1), held-out test split (3 patients, 540 slices).

**Prostate accuracy and clinical error modes**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Prostate Dice (all slices) | 0.873 ± 0.003 | 0.871 ± 0.008 | 0.885 | 0.867 |
| Prostate IoU | 0.775 ± 0.005 | 0.771 ± 0.013 | 0.794 | 0.765 |
| Prostate Dice: central slices | 0.906 ± 0.001 | 0.909 ± 0.004 | 0.918 | 0.904 |
| Prostate Dice: edge slices | 0.741 ± 0.016 | 0.713 ± 0.028 | 0.753 | 0.702 |
| Prostate boundary F-score (2 px) | 0.920 ± 0.006 | 0.915 ± 0.012 | 0.940 | 0.925 |
| Under-contouring (% of prostate missed) | 12.2 ± 2.5 | 12.7 ± 0.7 | 8.3 | 11.0 |
| Leak into bladder (% of prostate area) | 0.92 ± 0.38 | 1.17 ± 0.31 | 0.83 | 0.50 |
| Leak into rectum (% of prostate area) | 0.26 ± 0.34 | 0.16 ± 0.04 | 0.01 | 0.00 |
| False-positive prostate-free slices (%) | 0.6 ± 0.8 | 0.6 ± 0.3 | 0.6 | 0.6 |

**Review-band rule (flag the slices around each end of the predicted prostate extent)**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Slices flagged for manual review (%) | 13.3 ± 0.0 | 13.3 ± 0.0 | 13.3 | 13.3 |
| Missed prostate area caught by the flag (%) | 28.3 ± 4.4 | 36.0 ± 4.5 | 41.2 | 44.6 |
| False-positive area caught by the flag (%) | 18.8 ± 1.0 | 16.6 ± 1.4 | 15.9 | 11.9 |
| Whole-slice omissions (count) | 13.3 ± 2.3 | 17.7 ± 3.1 | 9.0 | 14.0 |
| Whole-slice omissions caught (%) | 95.1 ± 4.3 | 92.6 ± 1.9 | 88.9 | 71.4 |
| Prostate Dice on auto-drafted (unflagged) slices | 0.895 ± 0.002 | 0.895 ± 0.005 | 0.907 | 0.892 |

**Per-class Dice and IoU**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| body Dice | 0.980 ± 0.000 | 0.981 ± 0.000 | 0.980 | 0.974 |
| body IoU | 0.961 ± 0.001 | 0.963 ± 0.001 | 0.960 | 0.950 |
| bone Dice | 0.938 ± 0.000 | 0.940 ± 0.001 | 0.935 | 0.919 |
| bone IoU | 0.882 ± 0.001 | 0.887 ± 0.001 | 0.877 | 0.850 |
| bladder Dice | 0.964 ± 0.002 | 0.964 ± 0.001 | 0.961 | 0.940 |
| bladder IoU | 0.930 ± 0.004 | 0.931 ± 0.002 | 0.925 | 0.887 |
| rectum Dice | 0.852 ± 0.010 | 0.858 ± 0.020 | 0.851 | 0.831 |
| rectum IoU | 0.742 ± 0.016 | 0.752 ± 0.030 | 0.741 | 0.711 |
| prostate Dice | 0.873 ± 0.003 | 0.871 ± 0.008 | 0.885 | 0.867 |
| prostate IoU | 0.775 ± 0.005 | 0.771 ± 0.013 | 0.794 | 0.765 |

**Resource profile**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Parameters (M) | 7.76 ± 0.00 | 3.19 ± 0.00 | 9.46 | 9.46 |
| Peak training VRAM (GB) | 0.89 ± 0.00 | 0.67 ± 0.00 | 7.66 | 1.08 |
| Training time per epoch (s) | 34.5 ± 0.1 | 38.3 ± 0.3 | 25.6 | 5.9 |
| Total training time (min) | 29.2 ± 0.1 | 32.3 ± 0.2 | 43.5 | 20.9 |
| Inference per slice (ms; 3D: volume time / slices) | 1.08 ± 0.00 | 0.94 ± 0.02 | 1.56 | 0.39 |
| Inference per scan (s) | 0.07 ± 0.00 | 0.06 ± 0.00 | 0.09 | 0.02 |
| Peak inference VRAM (GB) | 0.97 ± 0.00 | 0.59 ± 0.00 | 2.06 | 0.37 |
| Latency for one input, batch of 1 (ms; 3D: one volume) | 2.62 ± 0.01 | 5.98 ± 0.03 | 55.33 | 7.59 |
