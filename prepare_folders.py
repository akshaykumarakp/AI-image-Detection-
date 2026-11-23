"""
Script to create sample dataset directories so you can drop your images right in.
Uses standard Python library (no dependencies required).
"""

import os
import struct

def make_minimal_bmp(filename, color_tuple):
    """Creates an uncompressed 24-bit 64x64 BMP image with standard library."""
    width, height = 64, 64
    row_bytes = width * 3
    padding = (4 - (row_bytes % 4)) % 4
    image_size = (row_bytes + padding) * height
    file_size = 54 + image_size

    header = struct.pack(
        '<2sIHHI',
        b'BM',
        file_size,
        0, 0,
        54
    )
    dib = struct.pack(
        '<IIIHHIIIIII',
        40,
        width, height,
        1,
        24,
        0,
        image_size,
        2835, 2835,
        0, 0
    )

    r, g, b = color_tuple
    row = (bytes([b, g, r]) * width) + (b'\x00' * padding)
    pixel_data = row * height

    with open(filename, 'wb') as f:
        f.write(header + dib + pixel_data)

def setup_dataset():
    dirs = [
        "dataset/train/real",
        "dataset/train/fake",
        "dataset/val/real",
        "dataset/val/fake",
        "test_samples"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Folders created successfully:")
    for d in dirs:
        print(f" [✓] {d}")

    # Generate quick sample placeholder images (BMP format works natively in PIL/PyTorch)
    for split in ["train", "val"]:
        count = 4 if split == "train" else 2
        for i in range(count):
            # Real placeholder: greenish tint
            make_minimal_bmp(f"dataset/{split}/real/sample_real_{i}.bmp", (30 + i * 20, 150, 50))
            # Fake/AI placeholder: reddish/magenta tint
            make_minimal_bmp(f"dataset/{split}/fake/sample_fake_{i}.bmp", (200, 40, 150 + i * 20))

    # Generate one test sample image
    make_minimal_bmp("test_samples/test_sample.bmp", (220, 30, 140))
    print("\n[✓] Generated sample placeholder images in dataset/ and test_samples/")

if __name__ == "__main__":
    setup_dataset()

