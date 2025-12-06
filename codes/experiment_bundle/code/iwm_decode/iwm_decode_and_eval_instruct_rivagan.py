import os
from glob import glob
from pathlib import Path
import csv

import cv2
import numpy as np
from imwatermark import WatermarkDecoder
from tqdm import tqdm  # ★ 新增：进度条

ROOT = r"C:\Users\Administrator\VINE"

# 1) 无攻击（clean）
ENC_DIR = os.path.join(
    ROOT,
    r"invisible_wm_exps\encoded_instruct_rivagan"
)

# 2) 经典 distortions_classic 攻击输出
CLASSIC_DIR = os.path.join(
    ROOT,
    r"output\distorted_rivagan_instruct_classic"
)

# 3) InstructPix2Pix 攻击输出根目录
INSTRUCT_DIR = os.path.join(
    ROOT,
    r"output\edited_rivagan_instructpix2pix"
)

wm_text = "qing"
wm_bytes_gt = wm_text.encode("utf-8")
wm_bits_len = len(wm_bytes_gt) * 8      # 32 bits
expected_bytes_len = len(wm_bytes_gt)   # 4 bytes

MAX_EVAL_IMAGES = 200  # ★ 只评估前 200 张，和 DWT/VINE 对齐


def normalize_decoded_bytes(raw, expected_len):
    if isinstance(raw, (bytes, bytearray)):
        b = bytes(raw)
    else:
        arr = np.array(raw, dtype=np.uint8).flatten()
        b = arr.tobytes()
    return b[:expected_len]


def decode_dir(dir_path, desc, scenario_key, writer, recursive=False, pattern="*.png"):
    """
    dir_path: 根目录
    desc: 打印用描述
    scenario_key: 写入 CSV 时标记是哪个场景（clean/classic/instruct）
    writer: csv.writer 对象
    recursive: 是否递归子目录（InstructPix2Pix 需要）
    pattern: '*.png' 或 '*_wm.png' 等
    """
    decoder = WatermarkDecoder("bytes", wm_bits_len)

    base = Path(dir_path)
    if recursive:
        paths = sorted(p for p in base.rglob(pattern) if p.is_file())
    else:
        paths = sorted(p for p in base.glob(pattern) if p.is_file())

    if MAX_EVAL_IMAGES is not None:
        paths = paths[:MAX_EVAL_IMAGES]

    if not paths:
        print(f"[WARN] No images in {dir_path}")
        return

    correct = 0
    total = 0

    print(f"\n[DECODE] {desc}  (dir={dir_path}, eval_first={len(paths)})")

    # ★ tqdm 进度条
    for idx, p in enumerate(tqdm(paths, desc=f"{scenario_key}", ncols=80)):
        p_str = str(p)
        bgr = cv2.imread(p_str, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [SKIP] read failed: {p_str}")
            continue

        try:
            raw = decoder.decode(bgr, "rivaGan")
            wm_bytes = normalize_decoded_bytes(raw, expected_bytes_len)
            total += 1
            is_correct = (wm_bytes == wm_bytes_gt)
            if is_correct:
                correct += 1

            rel_path = str(p.relative_to(base))

            # 前三张额外打印字符串形式
            if idx < 3:
                try:
                    decoded_str = wm_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    decoded_str = "<decode-error>"
                print(
                    f"  [{idx+1}] file={rel_path} "
                    f"decoded='{decoded_str}' correct={is_correct}"
                )

            # ★ 写入 CSV
            writer.writerow([
                scenario_key,
                rel_path,
                os.path.basename(p_str),
                is_correct
            ])

        except Exception as e:
            print(f"  [ERR] {p_str}: {e}")

    if total > 0:
        acc = correct / total
        print(f"[RESULT] {desc}: {correct}/{total} correct, accuracy={acc:.4f}")
    else:
        print(f"[RESULT] {desc}: no valid images.")


def main():
    print("[DEBUG] Using iwm_decode_and_eval_instruct_rivagan (clean + classic + instruct, CSV + tqdm)")

    # 必须先加载 RivaGAN 模型
    WatermarkDecoder.loadModel()

    # ★ 准备 CSV 输出
    csv_path = os.path.join(ROOT, "rivagan_instruct_eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 表头：场景、相对路径、文件名、是否解码成功
        writer.writerow(["scenario", "rel_path", "filename", "is_correct"])

        # 1) 无攻击：encoded_instruct_rivagan 下的 *_wm.png
        decode_dir(
            ENC_DIR,
            "Clean encoded (RivaGAN on INSTRUCT)",
            scenario_key="clean",
            writer=writer,
            recursive=False,
            pattern="*_wm.png"
        )

        # 2) 经典 distortions_classic 攻击：distorted_rivagan_instruct_classic 下的 *_wm.png
        decode_dir(
            CLASSIC_DIR,
            "Classic distortions (distortions_classic)",
            scenario_key="classic",
            writer=writer,
            recursive=False,
            pattern="*_wm.png"
        )

        # 3) InstructPix2Pix 攻击：edited_rivagan_instructpix2pix 下递归找所有 .png
        decode_dir(
            INSTRUCT_DIR,
            "InstructPix2Pix editing",
            scenario_key="instructpix2pix",
            writer=writer,
            recursive=True,
            pattern="*.png"
        )

    print(f"\n[DONE] RivaGAN evaluation on INSTRUCT-1K (first {MAX_EVAL_IMAGES} images per scenario) finished.")
    print(f"[INFO] CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
