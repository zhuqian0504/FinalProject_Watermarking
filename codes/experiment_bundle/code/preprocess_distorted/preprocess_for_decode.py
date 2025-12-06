import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess attacked images before decoding: "
            "center-crop + resize + denoise + deblock + brightness/contrast normalization."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root folder of attacked images (e.g., ./output/distortedADV_wmed_wbench_small_lambda1_0_v2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root folder to save preprocessed images "
             "(e.g., ./output/distortedADV_wmed_wbench_small_lambda1_0_v2_preproc)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target square size to resize images to (default: 512).",
    )
    return parser.parse_args()


def is_image_file(path: Path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """对任意长宽比图像做中心裁剪成正方形。"""
    w, h = img.size
    if w == h:
        return img
    if w > h:
        left = (w - h) // 2
        box = (left, 0, left + h, h)
    else:
        top = (h - w) // 2
        box = (0, top, w, top + w)
    return img.crop(box)


def preprocess_single_image(img: Image.Image, target_size: int) -> Image.Image:
    """
    解码前预处理流水线：
    1) 中心裁剪为正方形
    2) resize 到 target_size
    3) 在 Y 通道上做中值滤波 + 轻微高斯模糊
    4) Y 通道直方图均衡 (增强亮度/对比度)
    5) Unsharp Mask 轻微锐化
    """
    # 保证 RGB
    img = img.convert("RGB")

    # 1) 中心裁剪成正方形
    img = center_crop_to_square(img)

    # 2) resize 到统一尺寸
    img = img.resize((target_size, target_size), Image.BICUBIC)

    # 3) 转到 YCbCr，仅在 Y 上做处理
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    # 3.1) Y 通道中值滤波（去噪、减轻马赛克边）
    y = y.filter(ImageFilter.MedianFilter(size=3))

    # 3.2) Y 通道轻微高斯模糊（抹掉高频噪声/压缩块）
    y = y.filter(ImageFilter.GaussianBlur(radius=0.6))

    # 4) Y 通道直方图均衡（拉回正常亮度/对比度范围）
    y = ImageOps.equalize(y)

    # 5) 合并回 YCbCr -> RGB
    merged = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

    # 6) 轻微锐化（把被模糊的结构稍微拉清晰些）
    merged = merged.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))

    return merged


def main():
    args = parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if not in_root.exists():
        raise FileNotFoundError(f"Input folder not found: {in_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in in_root.rglob("*") if is_image_file(p)]
    print(f"Found {len(image_paths)} images in {in_root}")

    for img_path in tqdm(image_paths, desc="Preprocessing for decode"):
        rel_path = img_path.relative_to(in_root)
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            continue

        processed = preprocess_single_image(img, args.target_size)

        # 使用原始后缀保存，避免额外 JPEG 损伤
        processed.save(out_path)

    print(f"Done. Preprocessed images saved to: {out_root}")


if __name__ == "__main__":
    main()
