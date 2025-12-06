import json
import numpy as np
from pathlib import Path


def mean_bit_acc(json_path: Path):
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
        return float("nan"), 0

    return float(np.mean(vals)), len(vals)


def main():
    print("\n=======================================================")
    print(" compare_preproc_effect.py is running ")
    print("=======================================================\n")

    # 修改这里来切换要比较的 λ
    lam_str = "1_3"   # 例如 "0_7", "0_8", "0_9", "1_1", "1_2"

    # baseline (无预处理) 结果路径
    baseline_path = Path(
        f"./output/detection_results/"
        f"vine_r_small_lambda{lam_str}_distortADV_v2/wm_acc_dict.json"
    )

    # 预处理后解码的结果路径
    preproc_path = Path(
        f"./output/detection_results/"
        f"vine_r_small_lambda{lam_str}_distortADV_preproc_v2/wm_acc_dict.json"
    )

    print(f"Baseline path : {baseline_path}")
    print(f"Preproc  path : {preproc_path}\n")

    base_mean, base_n = mean_bit_acc(baseline_path)
    def_mean, def_n = mean_bit_acc(preproc_path)

    print(f"Baseline  (no preprocess): mean bit_acc = {base_mean:.6f}  | entries = {base_n}")
    print(f"Preprocess (with defense): mean bit_acc = {def_mean:.6f}  | entries = {def_n}")

    print("\n-------------------------------------------------------")
    if def_mean > base_mean:
        print("Result: ✅ Preprocessing IMPROVES ADV attack robustness.")
    else:
        print("Result: ❌ Preprocessing did NOT improve robustness (or not yet).")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()
