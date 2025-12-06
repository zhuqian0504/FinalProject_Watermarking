import json
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 配置部分
# =========================

# 所有要画的 lambda 值
lambda_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

# 把 float 的 lambda 映射到你目录里用的字符串后缀
# 例如 0.7 -> "0_7", 1.0 -> "1_0"
def format_lambda_suffix(lam: float) -> str:
    return str(lam).replace(".", "_")

# 经典攻击（classic）结果目录模板
CLASSIC_DIR_TEMPLATE = (
    "./output/detection_results/"
    "vine_r_small_lambda{lam}_distort_classic_v2/wm_acc_dict.json"
)

# 高级攻击（advanced）结果目录模板
ADV_DIR_TEMPLATE = (
    "./output/detection_results/"
    "vine_r_small_lambda{lam}_distortADV_v2/wm_acc_dict.json"
)

# 画质指标（PSNR / SSIM / LPIPS）——来自你已经跑出的结果
# 这里直接写死，避免再读文件
quality_metrics = {
    0.7: {"PSNR": 40.302731384991525, "SSIM": 0.9951207661012248, "LPIPS": 0.004375284374691546},
    0.8: {"PSNR": 39.21012858683765,  "SSIM": 0.9944177653781557, "LPIPS": 0.005176748303056229},
    0.9: {"PSNR": 38.23734592249654,  "SSIM": 0.9937064136305271, "LPIPS": 0.006031723682535812},
    1.0: {"PSNR": 37.37593887878522,  "SSIM": 0.9929883230225814, "LPIPS": 0.006927708258153871},
    1.1: {"PSNR": 36.60717365954726,  "SSIM": 0.9922655486890467, "LPIPS": 0.00785669280681759},
    1.2: {"PSNR": 35.91662197701183,  "SSIM": 0.9915401445372511, "LPIPS": 0.008825100196991115},
    1.3: {"PSNR": 35.29417080291303,  "SSIM": 0.9908120568200962, "LPIPS": 0.009818487698212267},
}


# =========================
# 2. 工具函数：从 wm_acc_dict.json 统计 bit_acc
# =========================

def load_mean_bit_acc(json_path: str):
    """
    从某个 wm_acc_dict.json 中读出所有 key 的 bit_acc 均值（整体均值）。
    返回 overall_mean_bit_acc。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    vals = []
    for key, stats in data.items():
        bit_acc_list = stats.get("bit_acc", [])
        if not bit_acc_list:
            continue
        v = bit_acc_list[0]
        vals.append(v)

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def collect_curve(json_template: str):
    """
    根据模板路径和 lambda 列表，返回每个 lambda 对应的 overall_mean_bit_acc。
    json_template 里用 {lam} 占位，例如:
    "./output/...lambda{lam}_distort_classic_v2/wm_acc_dict.json"
    """
    mean_bit_accs = []
    for lam in lambda_values:
        suffix = format_lambda_suffix(lam)
        path = json_template.format(lam=suffix)
        try:
            m = load_mean_bit_acc(path)
            mean_bit_accs.append(m)
            print(f"[INFO] λ={lam}: mean bit_acc = {m:.4f} from {path}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            mean_bit_accs.append(float("nan"))
    return np.array(mean_bit_accs)


# =========================
# 3. 收集所有曲线数据
# =========================

# 3.1 画质（直接从 quality_metrics 里取）
psnr_list = np.array([quality_metrics[lam]["PSNR"] for lam in lambda_values])
ssim_list = np.array([quality_metrics[lam]["SSIM"] for lam in lambda_values])
lpips_list = np.array([quality_metrics[lam]["LPIPS"] for lam in lambda_values])

# 3.2 鲁棒性：经典攻击 / 高级攻击
classic_mean_bit_acc = collect_curve(CLASSIC_DIR_TEMPLATE)
adv_mean_bit_acc = collect_curve(ADV_DIR_TEMPLATE)


# =========================
# 4. 画图 & 导出
# =========================

# 图像保存目录
os.makedirs("./figures", exist_ok=True)

# ------ 图1：λ vs 画质（PSNR & LPIPS）------
plt.figure(figsize=(6, 4))
plt.plot(lambda_values, psnr_list, marker="o", label="PSNR (dB)")
plt.xlabel("lambda_strength (λ)")
plt.ylabel("PSNR (dB)")
plt.title("Figure 1: λ vs PSNR")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fig1_lambda_vs_psnr.png", dpi=300)

plt.figure(figsize=(6, 4))
plt.plot(lambda_values, lpips_list, marker="o", label="LPIPS")
plt.xlabel("lambda_strength (λ)")
plt.ylabel("LPIPS (lower is better)")
plt.title("Figure 1b: λ vs LPIPS")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fig1b_lambda_vs_lpips.png", dpi=300)

# ------ 图2：λ vs mean bit_acc（经典攻击）------
plt.figure(figsize=(6, 4))
plt.plot(lambda_values, classic_mean_bit_acc, marker="o")
plt.xlabel("lambda_strength (λ)")
plt.ylabel("mean bit_acc (classic attacks)")
plt.ylim(0.0, 1.05)
plt.title("Figure 2: λ vs mean bit_acc (classic)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fig2_lambda_vs_bitacc_classic.png", dpi=300)

# ------ 图3：λ vs mean bit_acc（高级攻击）------
plt.figure(figsize=(6, 4))
plt.plot(lambda_values, adv_mean_bit_acc, marker="o")
plt.xlabel("lambda_strength (λ)")
plt.ylabel("mean bit_acc (advanced attacks)")
plt.ylim(0.0, 1.05)
plt.title("Figure 3: λ vs mean bit_acc (advanced)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fig3_lambda_vs_bitacc_advanced.png", dpi=300)

print("All figures saved under ./figures/")
