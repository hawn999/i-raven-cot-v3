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

import os
import json
import numpy as np
import xml.etree.ElementTree as ET

def inspect_sample(npz_path, black_threshold=5):
    base, _ = os.path.splitext(npz_path)
    xml_path = base + ".xml"

    print("=== Sample ===")
    print("npz:", npz_path)
    print("xml:", xml_path)

    # 1. 载入 npz 图像 + target/predict
    data = np.load(npz_path, allow_pickle=True)
    images = data["image"]   # 形状应该是 (22, H, W)
    target = int(data["target"])
    pred = int(data["predict"])
    print(f"target={target}, pred={pred}")

    # 2. 载入 xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    panel_elems = root.find("Panels").findall("Panel")
    print(f"Panels in xml: {len(panel_elems)}, images: {len(images)}")
    print()

    # 3. 逐 panel 调试
    for idx, (panel_img, panel_elem) in enumerate(zip(images, panel_elems)):
        mean_val = float(panel_img.mean())
        is_all_zero = not panel_img.any()
        is_black_like = mean_val < black_threshold

        # 计算这个 panel 在矩阵/答案中的位置（3×5 + 2×4）
        if idx < 14:
            # 上下文部分（3×5 展平去掉最后一格）
            # 原始 3×5 的 index = idx（0..13，对应 0..13；14 是缺的那格）
            row = idx // 5
            col = idx % 5
            kind = f"context(row={row}, col={col})"
        else:
            # 答案部分（2×4）
            ans_idx = idx - 14
            ans_row = ans_idx // 4
            ans_col = ans_idx % 4
            kind = f"candidate(row={ans_row}, col={ans_col}, idx={ans_idx})"

        print("=" * 60)
        print(f"Panel #{idx}  [{kind}]  mean={mean_val:.2f}  "
              f"all_zero={is_all_zero}  black_like={is_black_like}")

        struct = panel_elem.find("Struct")
        struct_name = struct.get("name") if struct is not None else "None"
        print(f"  Struct: {struct_name}")

        # 统计一下这个 panel 里的实体总数
        total_entities = 0

        for comp in struct.findall("Component"):
            comp_id = comp.get("id")
            comp_name = comp.get("name")
            layout = comp.find("Layout")
            if layout is None:
                print(f"    Component {comp_id} / {comp_name}: NO LAYOUT")
                continue

            num_level = layout.get("Number")  # level
            pos_slots = layout.get("Position")
            uniformity = layout.get("Uniformity")

            try:
                pos_slots_parsed = json.loads(pos_slots) if pos_slots is not None else None
            except Exception:
                pos_slots_parsed = pos_slots

            entities = layout.findall("Entity")
            total_entities += len(entities)

            print(f"  - Component {comp_id} / {comp_name}")
            print(f"      Layout   : {layout.get('name')}")
            print(f"      Number   : level={num_level}")
            print(f"      Position : slots={pos_slots_parsed}")
            print(f"      Uniformity: {uniformity}")
            print(f"      #Entities: {len(entities)}")

            for e_idx, ent in enumerate(entities):
                bbox = ent.get("bbox")
                t = ent.get("Type")
                s = ent.get("Size")
                c = ent.get("Color")
                ang = ent.get("Angle")
                print(f"        Entity {e_idx}: "
                      f"Type={t}, Size={s}, Color={c}, Angle={ang}, bbox={bbox}")

        print(f"  Total entities in this panel: {total_entities}")
        print()

if __name__ == "__main__":
    npz_path = "/path/to/RAVEN_XXXX_XXXX.npz"
    inspect_sample(npz_path)

