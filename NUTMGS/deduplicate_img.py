import os
from pathlib import Path
import hashlib
import shutil
import random

def hash_file(filepath, chunk_size=8192):
    """Generate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

base_dir = '/Users/Hugo/Coding_Projects/Sidd Lab/storage_bucket/rajasthan_top_10_images_for_label/images'
avoid_dupe = '/Users/Hugo/Coding_Projects/Sidd Lab/storage_bucket/rajasthan_top_10_images_for_label/new_images'
dest_dir = '/Users/Hugo/Coding_Projects/Sidd Lab/storage_bucket/rajasthan_top_10_images_for_label/newer_images'

os.makedirs(dest_dir, exist_ok=True)

already_found = set()

source_path = Path(base_dir)
avoid_path = Path(avoid_dupe)
dest_path = Path(dest_dir)

for img in os.listdir(avoid_path):
    img_hash = hash_file(source_path / img)
    already_found.add(img_hash)

num_removed = 0
for img in os.listdir(source_path):
    img_hash = hash_file(source_path / img)
    if img_hash in already_found:
        num_removed += 1
        continue

    already_found.add(img_hash)
    shutil.copy2(source_path / img, dest_path / img)

print(f'Number of duplicates removed: {num_removed}')