import os
import json

root = r".\output\detection_results\vine_r_instructPix2Pix_200"

print("Files in result folder:")
print(os.listdir(root))

path = os.path.join(root, "wm_acc_dict.json")

if not os.path.isfile(path):
    print("\n[ERROR] wm_acc_dict.json not found, please check the folder.")
    raise SystemExit

print(f"\n>>> Loading {path} ...")
with open(path, "r") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))

# 尝试把里面的结构打印出来，看看长什么样
for k, v in data.items():
    print("\nTask:", k)
    print("Type of value:", type(v))
    print("Value:", v)
