import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pywt

# ====== paths & config ======
ROOT = r"C:\Users\Administrator\VINE"

SRC_DIR = os.path.join(
    ROOT,
    r"W-Bench-INSTRUCT-only\INSTRUCT_1K\image"
)

OUT_DIR = os.path.join(
    ROOT,
    r"baseline_dwt_dct_instruct\INSTRUCT_1K\image"
)

os.makedirs(OUT_DIR, exist_ok=True)

MAX_IMAGES = None  # ★ 处理全部 INSTRUCT_1K（1000 张）；后面评估时再选前 200
ALPHA = 10.0       # 控制嵌入强度


def embed_bit_in_block(block, bit, alpha):
    dct = cv2.dct(block.astype(np.float32))
    i, j = 4, 4
    coef = dct[i, j]

    if bit == 1:
        if coef < 0:
            coef = -coef
        if abs(coef) < alpha:
            coef = alpha
    else:
        if coef > 0:
            coef = -coef
        if abs(coef) < alpha:
            coef = -alpha

    dct[i, j] = coef
    block_rec = cv2.idct(dct)
    return block_rec


def embed_watermark_y_channel(img_bgr, bit=1):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    coeffs2 = pywt.dwt2(Y.astype(np.float32), "haar")
    LL, (LH, HL, HH) = coeffs2

    h, w = LL.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    LL_crop = LL[:h8, :w8]

    for by in range(0, h8, 8):
        for bx in range(0, w8, 8):
            block = LL_crop[by:by+8, bx:bx+8]
            block_wm = embed_bit_in_block(block, bit, ALPHA)
            LL_crop[by:by+8, bx:bx+8] = block_wm

    LL[:h8, :w8] = LL_crop
    Y_rec = pywt.idwt2((LL, (LH, HL, HH)), "haar")
    Y_rec = np.clip(Y_rec, 0, 255).astype(np.uint8)

    ycrcb_rec = cv2.merge([Y_rec, Cr, Cb])
    bgr_rec = cv2.cvtColor(ycrcb_rec, cv2.COLOR_YCrCb2BGR)
    return bgr_rec


def main():
    img_paths = sorted(glob(os.path.join(SRC_DIR, "*.png")))
    if MAX_IMAGES is not None:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"[INFO] Found {len(img_paths)} images in {SRC_DIR}")
    if not img_paths:
        print("[WARN] no images to process.")
        return

    for idx, p in enumerate(img_paths):
        name = os.path.basename(p)
        print(f"[EMBED] {idx+1}/{len(img_paths)}  {name}")

        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [SKIP] failed to read {p}")
            continue

        bgr_wm = embed_watermark_y_channel(bgr, bit=1)

        # ★ 关键改动：和官方 VINE 一样，用 *_wm.png 结尾
        # 例如 0_248357.png -> 0_248357_wm.png
        out_name = Path(name).stem + "_wm.png"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, bgr_wm)
        print(f"  [SAVE] {out_path}")

    print("[DONE] DWT+DCT embedding on INSTRUCT_1K (ALL images) finished.")


if __name__ == "__main__":
    main()
