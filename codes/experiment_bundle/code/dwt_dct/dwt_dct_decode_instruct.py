import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pywt
import csv

ROOT = r"C:\Users\Administrator\VINE"

# 根据需要切换：
#   无攻击： r"baseline_dwt_dct_instruct\INSTRUCT_1K\image"
#   经典攻击： r"output\distorted_dwt_dct_instruct_classic"
#   InstructPix2Pix： r"output\edited_dwt_dct_instructpix2pix"
WM_DIR = os.path.join(
    ROOT,
    r"output\edited_dwt_dct_instructpix2pix"   # ★ 现在做的是 InstructPix2Pix 攻击
)

MAX_IMAGES = 200  # 只评估前 200 张


def decode_bit_from_image(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    coeffs2 = pywt.dwt2(Y.astype(np.float32), "haar")
    LL, (LH, HL, HH) = coeffs2

    h, w = LL.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    LL_crop = LL[:h8, :w8]

    ones = 0
    zeros = 0

    for by in range(0, h8, 8):
        for bx in range(0, w8, 8):
            block = LL_crop[by:by+8, bx:bx+8]
            dct = cv2.dct(block.astype(np.float32))
            coef = dct[4, 4]
            if coef > 0:
                ones += 1
            else:
                zeros += 1

    total_blocks = ones + zeros
    ratio_ones = ones / total_blocks if total_blocks > 0 else 0.0
    decoded_bit = 1 if ones >= zeros else 0
    return decoded_bit, ratio_ones, total_blocks


def main():
    wm_path = Path(WM_DIR)

    # ★ 递归地找所有 png：INSTRUCT_Pix2Pix/5/*.png 也能找到
    all_pngs = sorted(wm_path.rglob("*.png"))

    if MAX_IMAGES is not None:
        img_paths = all_pngs[:MAX_IMAGES]
    else:
        img_paths = all_pngs

    print(f"[INFO] Found {len(img_paths)} watermarked images (selected) under {WM_DIR}")
    if not img_paths:
        print("[WARN] no images to decode.")
        return

    correct = 0
    expected_bit = 1

    exp_name = wm_path.name  # 例如 edited_dwt_dct_instructpix2pix
    csv_filename = f"dwt_dct_decode_results_{exp_name}.csv"
    csv_path = os.path.join(ROOT, csv_filename)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "rel_path",
            "total_blocks",
            "ratio_ones",
            "decoded_bit",
            "expected_bit",
            "is_correct"
        ])

        for idx, p in enumerate(img_paths):
            p = Path(p)
            name = p.name
            rel_path = str(p.relative_to(wm_path))

            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[SKIP] failed to read {p}")
                continue

            decoded_bit, ratio_ones, total_blocks = decode_bit_from_image(bgr)
            is_correct = (decoded_bit == expected_bit)
            if is_correct:
                correct += 1

            print(
                f"图像: {rel_path}\n"
                f"  block 数量: {total_blocks}\n"
                f"  解码为 1 的比例: {ratio_ones:.4f}\n"
                f"  多数投票结果 decoded_bit = {decoded_bit} (期望 {expected_bit})  "
                f"{'-> OK' if is_correct else '-> FAIL'}\n"
            )

            writer.writerow([
                name,
                rel_path,
                int(total_blocks),
                float(ratio_ones),
                int(decoded_bit),
                int(expected_bit),
                int(is_correct)
            ])

    total_images = len(img_paths)
    accuracy = correct / total_images if total_images > 0 else 0.0

    print(f"总计 {total_images} 张图，decoded_bit == {expected_bit} 的有 {correct} 张。")
    print(f"整体准确率 accuracy = {accuracy:.4f}")
    print(f"[INFO] CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
