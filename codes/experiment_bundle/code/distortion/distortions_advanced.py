import argparse
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply advanced traditional attacks (resize+crop, color jitter, "
            "blur, noise, occlusion, jpeg) to a folder of watermarked images, "
            "preserving folder structure."
        )
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
        help="Root folder to save ADVANCED distorted images "
             "(e.g., ./output/distortedADV_wmed_wbench_small_lambda1_0_v2)",
    )

    # 基础亮度/对比度
    parser.add_argument("--brightness_min", type=float, default=0.9)
    parser.add_argument("--brightness_max", type=float, default=1.1)
    parser.add_argument("--contrast_min", type=float, default=0.9)
    parser.add_argument("--contrast_max", type=float, default=1.1)

    # 裁剪缩放
    parser.add_argument(
        "--crop_scale_min", type=float, default=0.85,
        help="Minimum random crop scale relative to original size (default: 0.85)",
    )
    parser.add_argument(
        "--crop_scale_max", type=float, default=1.0,
        help="Maximum random crop scale (default: 1.0)",
    )

    # 颜色抖动（饱和度）
    parser.add_argument(
        "--saturation_min", type=float, default=0.9,
        help="Lower bound of random saturation factor (default: 0.9)",
    )
    parser.add_argument(
        "--saturation_max", type=float, default=1.1,
        help="Upper bound of random saturation factor (default: 1.1)",
    )

    # 模糊 + 噪声 + JPEG
    parser.add_argument("--blur_radius", type=float, default=1.0)
    parser.add_argument(
        "--noise_std", type=float, default=0.02,
        help="Std of Gaussian noise in [0,1] range (default: 0.02)",
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=60,
        help="JPEG quality simulating compression attack (default: 60)",
    )

    # 遮挡 / 马赛克参数
    parser.add_argument(
        "--occlusion_prob", type=float, default=0.5,
        help="Probability to apply a random occlusion/mosaic patch (default: 0.5)",
    )
    parser.add_argument(
        "--occ_scale_min", type=float, default=0.1,
        help="Minimum side length as fraction of image min(w,h) (default: 0.1)",
    )
    parser.add_argument(
        "--occ_scale_max", type=float, default=0.25,
        help="Maximum side length as fraction of image min(w,h) (default: 0.25)",
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def random_resized_crop(img: Image.Image,
                        rng: np.random.Generator,
                        scale_min: float,
                        scale_max: float) -> Image.Image:
    """随机裁剪一块区域再 resize 回原尺寸。"""
    img = img.convert("RGB")
    w, h = img.size
    scale = float(rng.uniform(scale_min, scale_max))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if new_w == w and new_h == h:
        return img

    left = int(rng.integers(0, max(1, w - new_w + 1)))
    top = int(rng.integers(0, max(1, h - new_h + 1)))
    crop_box = (left, top, left + new_w, top + new_h)
    cropped = img.crop(crop_box)
    resized = cropped.resize((w, h), Image.BICUBIC)
    return resized


def apply_occlusion_mosaic(img: Image.Image,
                           rng: np.random.Generator,
                           occ_scale_min: float,
                           occ_scale_max: float) -> Image.Image:
    """在图像上随机打一块遮挡/马赛克。"""
    img = img.convert("RGB")
    w, h = img.size
    min_side = min(w, h)

    scale = float(rng.uniform(occ_scale_min, occ_scale_max))
    patch_size = max(4, int(min_side * scale))

    if patch_size >= min_side:
        return img

    left = int(rng.integers(0, w - patch_size + 1))
    top = int(rng.integers(0, h - patch_size + 1))
    box = (left, top, left + patch_size, top + patch_size)

    patch = img.crop(box)
    # 做个马赛克：先缩小再放大
    small = patch.resize((4, 4), Image.BILINEAR)
    mosaic = small.resize((patch_size, patch_size), Image.NEAREST)

    img.paste(mosaic, box)
    return img


def apply_advanced_distortions(
    img: Image.Image,
    rng: np.random.Generator,
    brightness_min: float,
    brightness_max: float,
    contrast_min: float,
    contrast_max: float,
    crop_scale_min: float,
    crop_scale_max: float,
    saturation_min: float,
    saturation_max: float,
    blur_radius: float,
    noise_std: float,
    jpeg_quality: int,
    occlusion_prob: float,
    occ_scale_min: float,
    occ_scale_max: float,
) -> Image.Image:

    img = img.convert("RGB")

    # 1) 随机裁剪 + 缩放回原尺寸
    img = random_resized_crop(img, rng, crop_scale_min, crop_scale_max)

    # 2) 亮度 / 对比度
    b_factor = float(rng.uniform(brightness_min, brightness_max))
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    c_factor = float(rng.uniform(contrast_min, contrast_max))
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 3) 颜色抖动（饱和度）
    s_factor = float(rng.uniform(saturation_min, saturation_max))
    img = ImageEnhance.Color(img).enhance(s_factor)

    # 4) 轻度模糊
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 5) 高斯噪声
    arr = np.asarray(img).astype(np.float32) / 255.0
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
        arr = arr + noise
        arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))

    # 6) 随机遮挡 / 马赛克（有一定概率触发）
    if rng.random() < occlusion_prob:
        img = apply_occlusion_mosaic(img, rng, occ_scale_min, occ_scale_max)

    # 7) JPEG 压缩伪影
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

    for img_path in tqdm(image_paths, desc="Processing ADVANCED distortions"):
        rel_path = img_path.relative_to(wm_root)
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            continue

        distorted = apply_advanced_distortions(
            img,
            rng=rng,
            brightness_min=args.brightness_min,
            brightness_max=args.brightness_max,
            contrast_min=args.contrast_min,
            contrast_max=args.contrast_max,
            crop_scale_min=args.crop_scale_min,
            crop_scale_max=args.crop_scale_max,
            saturation_min=args.saturation_min,
            saturation_max=args.saturation_max,
            blur_radius=args.blur_radius,
            noise_std=args.noise_std,
            jpeg_quality=args.jpeg_quality,
            occlusion_prob=args.occlusion_prob,
            occ_scale_min=args.occ_scale_min,
            occ_scale_max=args.occ_scale_max,
        )

        distorted.save(out_path)


if __name__ == "__main__":
    main()
