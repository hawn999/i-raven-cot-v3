import os, glob, numpy as np

dataset_dir = "/home/scxhc1/nvme_data/cot_datasets/v2_test1/I-RAVEN"
pattern = "**/*_*.npz"

expected_panels = 22
expected_side = 160
expected_total = expected_panels * expected_side * expected_side

bad = []
for p in glob.glob(os.path.join(dataset_dir, pattern), recursive=True):
    try:
        with np.load(p) as d:
            sz = d["image"].size
        if sz != expected_total:
            bad.append((p, sz))
            print(p)
            # print(bad)
    except Exception as e:
        bad.append((p, f"error: {e}"))

print(f"总计发现 {len(bad)} 个不符合 22×{expected_side}×{expected_side} 的样本：")
for p, sz in bad[:50]:
    print(" -", p, sz)
if len(bad) > 50:
    print(" ... 仅显示前 50 条")
