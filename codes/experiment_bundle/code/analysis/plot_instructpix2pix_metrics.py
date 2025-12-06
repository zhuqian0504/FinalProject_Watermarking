import os
import json
import matplotlib.pyplot as plt

# 1. 结果目录
root = r".\output\detection_results\vine_r_instructPix2Pix_200"
json_path = os.path.join(root, "wm_acc_dict.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 取出 InstructPix2Pix 那一项
key = "INSTRUCT_Pix2Pix\\5"
metrics = data[key]

bit_acc = metrics["bit_acc"][0]
tpr_1 = metrics["TPR@1%FPR"][0]
tpr_01 = metrics["TPR@0.1%FPR"][0]
auroc = metrics["AUROC"][0]

names = ["bit_acc", "TPR@1%FPR", "TPR@0.1%FPR", "AUROC"]
values = [bit_acc, tpr_1, tpr_01, auroc]

print("InstructPix2Pix metrics:")
for n, v in zip(names, values):
    print(f"{n}: {v:.4f}")

# 3. 画图 + 数值标签
plt.figure(figsize=(6, 4))
plt.ylim(0.0, 1.05)

bars = plt.bar(names, values)
plt.ylabel("Score")
plt.title("VINE-R under InstructPix2Pix Attack")

# 在每个柱子上方标注数值
for bar, val in zip(bars, values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,            # 稍微高于柱顶
        f"{val:.4f}",             # 保留四位小数
        ha="center",
        va="bottom",
        fontsize=9
    )

# 4. 保存到 figures 目录
os.makedirs(r".\figures", exist_ok=True)
out_path = r".\figures\instructpix2pix_metrics.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"\nSaved figure to: {out_path}")
