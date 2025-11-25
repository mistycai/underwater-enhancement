# Multi-Stage Underwater Image Enhancement and Detection

This repository implements a modular, physically interpretable pipeline for underwater image enhancement and its impact on YOLOv8-based object detection.

Pipeline stages:
1. Red-channel compensation (illumination / color correction)
2. Luminance-space CLAHE with detection-aware gating (contrast enhancement)
3. Classical denoising and particle suppression
4. Multi-scale fusion of enhanced outputs
5. Evaluation with YOLOv8 on RUOD

Data is **not** stored in this repo. See `data/README.md` for dataset setup.

# Usage for Red Restoration

1. to evaluate generated results and save generated images, run:
```bash
python eval_rcc_wrcc.py \
    --val_dir ./data/val_pic \
    --save_imgs \
    --out_dir ./results/rcc_wrcc_images
```
2. just evaluate generated results, run:
```bash
python eval_rcc_wrcc.py --val_dir data/ruod/val/images
```

3. to visualize how the evaluation metrics (UIQM, UCIQE, and contrast gain) change with different alpha values, run:
```bash
python plot.py
```