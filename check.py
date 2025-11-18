# import os, glob, numpy as np

# dataset_dir = "/home/scxhc1/nvme_data/cot_datasets/v2_test1/I-RAVEN"
# pattern = "**/*_*.npz"

# expected_panels = 22
# expected_side = 160
# expected_total = expected_panels * expected_side * expected_side

# bad = []
# for p in glob.glob(os.path.join(dataset_dir, pattern), recursive=True):
#     try:
#         with np.load(p) as d:
#             sz = d["image"].size
#         if sz != expected_total:
#             bad.append((p, sz))
#             print(p)
#             # print(bad)
#     except Exception as e:
#         bad.append((p, f"error: {e}"))

# print(f"总计发现 {len(bad)} 个不符合 22×{expected_side}×{expected_side} 的样本：")
# for p, sz in bad[:50]:
#     print(" -", p, sz)
# if len(bad) > 50:
#     print(" ... 仅显示前 50 条")

import numpy as np
from PIL import Image

IMG_SIZE = 160

img = np.array(Image.open("/path/to/problem_000123.png").convert("L"))

# 先只看上面 3x5 的 15 个面板
panels = []
for r in range(3):
    for c in range(5):
        patch = img[r*IMG_SIZE:(r+1)*IMG_SIZE, c*IMG_SIZE:(c+1)*IMG_SIZE]
        panels.append(((r, c), patch))

for (r, c), patch in panels:
    # 全 0 或者 95%以上像素非常接近 0（黑）
    black_ratio = np.mean(patch < 10)
    if black_ratio > 0.95:
        print("黑块面板:", r, c, "黑像素比例:", black_ratio)
