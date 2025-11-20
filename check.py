# 7w 144 143 center
import os, glob, numpy as np

dataset_dir = "./test4_10000"
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

exit()


# debug_black_sample.py
# 用来检查某个 RAVEN 样本中每个 panel 的属性（Number / Position / Entities）

import os
import sys
import json
import numpy as np
import xml.etree.ElementTree as ET


def panel_kind(idx, n_rows=3, n_cols=5):
    """
    根据 3x5 + 8答案 的协议，返回这个 panel 是：
    - context(row=i, col=j)
    - candidate(row=i, col=j, idx=k)
    """
    n_context = n_rows * n_cols - 1  # 3*5-1 = 14

    if idx < n_context:
        row = idx // n_cols
        col = idx % n_cols
        return f"context(row={row}, col={col})"
    else:
        ans_idx = idx - n_context
        ans_row = ans_idx // 4
        ans_col = ans_idx % 4
        return f"candidate(row={ans_row}, col={ans_col}, idx={ans_idx})"


def inspect_sample(npz_path, black_threshold=5.0):
    base, _ = os.path.splitext(npz_path)
    xml_path = base + ".xml"

    if not os.path.exists(npz_path):
        print("npz not found:", npz_path)
        return
    if not os.path.exists(xml_path):
        print("xml not found:", xml_path)
        return

    print("=== Sample ===")
    print("npz:", npz_path)
    print("xml:", xml_path)
    print()

    # 1. 载入 npz
    data = np.load(npz_path, allow_pickle=True)
    images = data["image"]  # (22, H, W)
    target = int(data.get("target", -1))
    pred = int(data.get("predict", -1))
    print(f"target = {target}, pred = {pred}")
    print(f"#panels in npz = {len(images)}")

    # 2. 载入 xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    panels_elem = root.find("Panels")
    if panels_elem is None:
        print("No <Panels> in xml")
        return
    panel_elems = panels_elem.findall("Panel")
    print(f"#panels in xml = {len(panel_elems)}")
    print()

    if len(panel_elems) != len(images):
        print("[WARN] #panels mismatch: xml has", len(panel_elems),
              "npz has", len(images))

    # 3. 统计 Number level 与实体个数的分布（全局）
    level_stats = {}  # level -> list of #entities

    # 4. 逐 panel 打印详细信息
    for idx, (panel_img, panel_elem) in enumerate(zip(images, panel_elems)):
        mean_val = float(panel_img.mean())
        is_all_zero = not panel_img.any()
        is_black_like = mean_val < black_threshold

        kind = panel_kind(idx)

        print("=" * 70)
        print(f"Panel #{idx}  [{kind}]")
        print(f"  mean_pixel={mean_val:.2f}, all_zero={is_all_zero}, black_like={is_black_like}")

        struct = panel_elem.find("Struct")
        struct_name = struct.get("name") if struct is not None else "None"
        print(f"  Struct: {struct_name}")

        total_entities = 0

        if struct is None:
            print("  [WARN] No <Struct> in this panel")
            continue

        for comp in struct.findall("Component"):
            comp_id = comp.get("id")
            comp_name = comp.get("name")

            layout = comp.find("Layout")
            if layout is None:
                print(f"    Component {comp_id} / {comp_name}: NO <Layout>")
                continue

            # 注意：你的 xml 里原始只存了 Number (level) 和 Position（全部槽位）
            # 如果你之后按建议增加 Number_value / Position_idx / Position_selected 也可以一起打印
            num_level_str = layout.get("Number", "0")
            try:
                num_level = int(num_level_str)
            except Exception:
                num_level = 0

            # 假设 Number.value = level + 1（由你观察到的 4图 -> level=3 推断）
            num_value = num_level + 1

            pos_slots = layout.get("Position_slots") or layout.get("Position")  # 兼容老字段
            try:
                pos_slots_parsed = json.loads(pos_slots) if pos_slots is not None else None
            except Exception:
                pos_slots_parsed = pos_slots

            # 如果你在 serialize 里加了 Position_idx/Position_selected，可以一起打印：
            pos_idx = layout.get("Position_idx")
            pos_sel = layout.get("Position_selected")

            entities = layout.findall("Entity")
            ent_count = len(entities)
            total_entities += ent_count

            level_stats.setdefault(num_level, []).append(ent_count)

            print(f"  - Component {comp_id} / {comp_name}")
            print(f"      Layout_name     : {layout.get('name')}")
            print(f"      Number_level    : {num_level}  ->  Number_value ~= {num_value}")
            print(f"      Position_slots  : {pos_slots_parsed}")
            print(f"      Position_idx    : {pos_idx}")
            print(f"      #entities       : {ent_count}")

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

    # 5. 打印 Number level 与实体个数的全局分布
    print("\n=== Number level vs #entities stats (global) ===")
    for lvl in sorted(level_stats.keys()):
        cnts = level_stats[lvl]
        uniq = sorted(set(cnts))
        print(f"  level={lvl}: #entities values = {uniq}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_black_sample.py /path/to/RAVEN_xxx_xxx.npz")
        sys.exit(1)
    npz_path = sys.argv[1]
    inspect_sample(npz_path)

# npz_path = "/home/scxhc1/nvme_data/cot_datasets/v2_test1/I-RAVEN/distribute_four/RAVEN_2224_train.npz"

