import os
from glob import glob
from pathlib import Path

import cv2
from imwatermark import WatermarkEncoder

ROOT = r"C:\Users\Administrator\VINE"

SRC_DIR = os.path.join(
    ROOT,
    r"W-Bench-INSTRUCT-only\INSTRUCT_1K\image"
)

OUT_DIR = os.path.join(
    ROOT,
    r"invisible_wm_exps\encoded_instruct_rivagan"
)

os.makedirs(OUT_DIR, exist_ok=True)

MAX_IMAGES = None  # 处理全部 1000 张

# ★ RivaGAN 配置：4 字符，32 bits
wm_text = "qing"
wm_bytes = wm_text.encode("utf-8")


def main():
    img_paths = sorted(glob(os.path.join(SRC_DIR, "*.png")))
    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"[INFO] Using {len(img_paths)} images for RivaGAN encoding (INSTRUCT_1K).")

    # 必须先加载 RivaGAN 模型
    WatermarkEncoder.loadModel()

    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", wm_bytes)

    for idx, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        stem = Path(img_name).stem
        out_name = stem + "_wm.png"
        out_path = Path(OUT_DIR) / out_name

        # ★ 如果已经有输出文件，则跳过（支持断点续跑）
        if out_path.exists():
            print(f"[SKIP] {idx+1}/{len(img_paths)} {img_name} -> exists, skip")
            continue

        print(f"[ENC] {idx+1}/{len(img_paths)}  {img_name}")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [SKIP] Failed to read {img_path}")
            continue

        # 使用 RivaGAN 模式
        bgr_wm = encoder.encode(bgr, "rivaGan")

        # 保持 *_wm.png 命名，兼容官方 InstructPix2Pix 逻辑
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr_wm)
        print(f"  [SAVE] {out_path}")

    print("[DONE] RivaGAN encoding on ALL INSTRUCT_1K images finished (skipped existing files).")


if __name__ == "__main__":
    main()
