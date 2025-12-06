import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# 要对比的 lambda 列表
LAMBDA_VALUES = [0.7, 1.0, 1.3]


def lam_to_suffix(lam: float) -> str:
    """把 0.7 -> '0_7' 这种格式，和你目录名一致。"""
    return str(lam).replace(".", "_")


def load_mean_bit_acc(json_path: Path):
    """从 wm_acc_dict.json 中取出所有 key 的 bit_acc 整体平均。"""
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    vals = []
    for key, stats in data.items():
        bit_list = stats.get("bit_acc", [])
        if bit_list:
            vals.append(float(bit_list[0]))

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main():
    baseline_means = []
    preproc_means = []

    print("\n==============================================")
    print("  Collecting baseline vs preproc (ADV attack)")
    print("==============================================\n")

    for lam in LAMBDA_VALUES:
        suf = lam_to_suffix(lam)

        base_path = Path(
            f"./output/detection_results/"
            f"vine_r_small_lambda{suf}_distortADV_v2/wm_acc_dict.json"
        )
        pre_path = Path(
            f"./output/detection_results/"
            f"vine_r_small_lambda{suf}_distortADV_preproc_v2/wm_acc_dict.json"
        )

        base_mean = load_mean_bit_acc(base_path)
        pre_mean = load_mean_bit_acc(pre_path)

        baseline_means.append(base_mean)
        preproc_means.append(pre_mean)

        print(
            f"λ = {lam:.1f} | baseline = {base_mean:.6f}, "
            f"preproc = {pre_mean:.6f}, diff = {pre_mean - base_mean:+.6f}"
        )

    baseline_means = np.array(baseline_means)
    preproc_means = np.array(preproc_means)

    # 确保有图目录
    os.makedirs("./figures", exist_ok=True)

    # ===== 关键改动：让两条线稍微分开 & 使用不同虚线样式 =====

    # 在 x 轴上做一点点水平偏移，避免点完全重合
    offset = 0.01
    x_base = [lam - offset for lam in LAMBDA_VALUES]
    x_pre  = [lam + offset for lam in LAMBDA_VALUES]

    plt.figure(figsize=(6, 4))

    # baseline：短虚线 + 圆点
    plt.plot(
        x_base,
        baseline_means,
        linestyle="--",
        marker="o",
        label="Baseline (no preprocess)",
    )

    # preproc：点划线 + 方块
    plt.plot(
        x_pre,
        preproc_means,
        linestyle=":",
        marker="s",
        label="Preprocess defense",
    )

    plt.xlabel("lambda_strength (λ)")
    plt.ylabel("mean bit_acc under ADV attack")
    plt.ylim(0.0, 1.05)
    plt.title("Effect of preprocessing under advanced attacks")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = "./figures/preproc_vs_baseline_adv_bitacc.png"
    plt.savefig(out_path, dpi=300)

    print("\nSaved figure to:", out_path)
    print("Done.\n")


if __name__ == "__main__":
    main()
