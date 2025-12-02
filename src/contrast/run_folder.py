# src/contrast/run_folder.py
import os
from pathlib import Path
import cv2

from .pipeline import apply_clahe_to_bgr


def list_images(folder):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    paths = []
    for name in os.listdir(folder):
        if any(name.lower().endswith(e) for e in exts):
            paths.append(Path(folder) / name)
    return sorted(paths)


def process_folder(input_dir, output_dir, use_gating=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list_images(input_dir)
    if not img_paths:
        print("[WARN] no images found in", input_dir)
        return

    count_apply = 0
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print("[WARN] cannot read", p)
            continue

        bgr_out, info = apply_clahe_to_bgr(bgr, use_gating=use_gating)
        if info["applied"]:
            count_apply += 1

        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), bgr_out)

    print(f"[INFO] processed {len(img_paths)} images, CLAHE applied to {count_apply}.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 -m src.contrast.run_folder <input_dir> <output_dir>")
    else:
        process_folder(sys.argv[1], sys.argv[2], use_gating=True)