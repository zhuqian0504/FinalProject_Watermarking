import argparse
import os
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply classic/traditional distortion attacks (brightness, contrast, blur, noise, jpeg) "
                    "to a folder of watermarked images, preserving folder structure."
    )
    parser.add_argument(
        "--wm_images_folder",
        type=str,
        required=True,
        help="Root folder of watermarked images (e.g., ./vine_encoded_wbench_small_lambda1_0_v2)",
    )
    parser.add_argument(
        "--edited_output_folder",
        type=str,
        required=True,
        help="Root folder to save distorted images (e.g., ./output/distorted_wmed_wbench_small_lambda1_0_v2)",
    )
    parser.add_argument(
        "--brightness_min",
        type=float,
        default=0.85,
        help="Lower bound of random brightness factor (default: 0.85)",
    )
    parser.add_argument(
        "--brightness_max",
        type=float,
        default=1.15,
        help="Upper bound of random brightness factor (default: 1.15)",
    )
    parser.add_argument(
        "--contrast_min",
        type=float,
        default=0.85,
        help="Lower bound of random contrast factor (default: 0.85)",
    )
    parser.add_argument(
        "--contrast_max",
        type=float,
        default=1.15,
        help="Upper bound of random contrast factor (default: 1.15)",
    )
    parser.add_argument(
        "--blur_radius",
        type=float,
        default=0.8,
        help="Gaussian blur radius (default: 0.8)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="Std of Gaussian noise in [0,1] (default: 0.02)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=65,
        help="JPEG quality simulating compression attack (default: 65)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_classic_distortions(img: Image.Image,
                              rng: np.random.Generator,
                              brightness_min: float,
                              brightness_max: float,
                              contrast_min: float,
                              contrast_max: float,
                              blur_radius: float,
                              noise_std: float,
                              jpeg_quality: int) -> Image.Image:

    img = img.convert("RGB")

    # ----- Brightness -----
    b_factor = rng.uniform(brightness_min, brightness_max)
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    # ----- Contrast -----
    c_factor = rng.uniform(contrast_min, contrast_max)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # ----- Gaussian Blur -----
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ----- Gaussian Noise -----
    arr = np.asarray(img).astype(np.float32) / 255.0
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
        arr = arr + noise
        arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))

    # ----- JPEG Compression Artifact -----
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")

    return img


def main():
    args = parse_args()

    wm_root = Path(args.wm_images_folder)
    out_root = Path(args.edited_output_folder)

    if not wm_root.exists():
        raise FileNotFoundError(f"Input folder not found: {wm_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in wm_root.rglob("*") if is_image_file(p)]
    print(f"Found {len(image_paths)} images in {wm_root}")

    rng = np.random.default_rng(args.seed)

    for img_path in tqdm(image_paths, desc="Processing traditional distortions"):
        rel_path = img_path.relative_to(wm_root)
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            continue

        distorted = apply_classic_distortions(
            img,
            rng=rng,
            brightness_min=args.brightness_min,
            brightness_max=args.brightness_max,
            contrast_min=args.contrast_min,
            contrast_max=args.contrast_max,
            blur_radius=args.blur_radius,
            noise_std=args.noise_std,
            jpeg_quality=args.jpeg_quality,
        )

        distorted.save(out_path)


if __name__ == "__main__":
    main()
