# Multi-Stage Underwater Image Enhancement and Detection

This repository implements a modular, physically interpretable pipeline for underwater image enhancement and its impact on YOLOv8 object detection.

Pipeline stages:
1. Red-channel compensation;
2. Luminance-space CLAHE with detection-aware gating;
3. Classical denoising and particle suppression;
4. Multi-scale fusion of enhanced outputs;
5. Evaluation with YOLOv8.

## Dataset
### Testing/validation data
- RUOD dataset: https://drive.google.com/file/d/1hxtbdgfVveUm_DJk5QXkNLokSCTa_E5o/view?usp=drive_link
- Our collected underwater dataset: https://drive.google.com/drive/folders/1lySfhvL3oNrMKPOJ3ja_Rih4kEU7aNd_?usp=sharing

## Pretrained model
- YOLOv8 checkpoint: https://drive.google.com/file/d/1mZJBh5zhojruV3r6U1PUEBfudPjdOqXi/view?usp=sharing

# Quick demo
Run the full fusion demo on a RUOD sample:
```bash
python3 ./src/run_fusion_demo.py ./data/RUOD/RUOD_pic/train/000015.jpg -o demo/ --layout horizontal \
    --rcc-alpha 0.5 \
    --clahe-clip 2.0 \
    --clahe-tile 16
```