import os
import random
import shutil

src_dir = "/Users/zjy/Downloads/RUOD/RUOD_pic/train"
dst_dir = "/Users/zjy/Downloads/RUOD/RUOD_1000"

os.makedirs(dst_dir, exist_ok=True)

all_files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

selected_files = random.sample(all_files, 1000)

for fname in selected_files:
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    shutil.copy(src_path, dst_path)

print("1000 images have been saved to:", dst_dir)
