# Multi-Stage Underwater Image Enhancement and Detection

This repository implements a modular, physically interpretable pipeline for underwater image enhancement and its impact on YOLOv8-based object detection.

Pipeline stages:
1. Red-channel compensation (illumination / color correction)
2. Luminance-space CLAHE with detection-aware gating (contrast enhancement)
3. Classical denoising and particle suppression
4. Multi-scale fusion of enhanced outputs
5. Evaluation with YOLOv8 on RUOD

Data is **not** stored in this repo. See `data/README.md` for dataset setup.
