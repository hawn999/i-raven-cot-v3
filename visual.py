import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import argparse
import os
import xml.etree.ElementTree as ET

from sympy.core.parameters import distribute


def parse_rules_from_xml(xml_path):
    """
    从 .xml 文件中解析规则信息
    返回一个字典，键是列索引 (str)，值是该列的规则字符串列表
    """
    if not os.path.exists(xml_path):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    rules_data = {}
    rules_root = root.find("Rules")
    if rules_root is None:
        return {}

    # 遍历 <Column_Rule_Set> (t=2, 3, 4)
    for col_rule_set in rules_root.findall("Column_Rule_Set"):
        col_idx = col_rule_set.get("column_index")
        col_rules_str = []

        # 遍历 <Component_Rule_Group> (C0, C1, ...)
        for comp_rule_group in col_rule_set.findall("Component_Rule_Group"):
            comp_id = comp_rule_group.get("component_id")

            # 遍历 <Rule>
            for rule in comp_rule_group.findall("Rule"):
                name = rule.get("name")
                attr = rule.get("attr")
                value = rule.get("value", "")  # 获取 value，如果不存在则为空

                # 缩写
                if name == "Progression":
                    name = "Prog"
                elif name == "Constant":
                    name = "Const"
                elif name == "Arithmetic":
                    name = "Arith"
                elif name == "Distribute_Three":
                    name = "Dist3"

                if "Number/Position" in attr:
                    attr = "Num/Pos"

                # 格式化字符串
                if value and value != "0":  # 只显示非零值
                    rule_str = f"C{comp_id}: {name}({attr}, {value})"
                else:
                    rule_str = f"C{comp_id}: {name}({attr})"
                col_rules_str.append(rule_str)

        rules_data[col_idx] = col_rules_str

    return rules_data


def build_rules_text(rules_info, n_cols=5):
    """
    将 rules_info（按列索引）整理为右侧竖栏要显示的文本。
    优先按列号从小到大；仅显示存在规则的列。
    """
    if not rules_info:
        return "no rule in XML"

    # 将 key 转成 int 后排序，保证顺序稳定
    try:
        sorted_cols = sorted(int(k) for k in rules_info.keys())
    except Exception:
        # 如果 key 不是纯数字，直接按字符串排序
        sorted_cols = sorted(rules_info.keys())

    sections = []
    for col in sorted_cols:
        key = str(col)
        if key not in rules_info or len(rules_info[key]) == 0:
            continue
        # 标题：列号（与上文一致，0-based）
        header = f"Col {col}"
        lines = [f"- {s}" for s in rules_info[key]]
        sections.append(header + "\n" + "\n".join(lines))

    if not sections:
        return "no visible rules"

    # 段落之间空一行
    return "\n\n".join(sections)


def main():
    # parser = argparse.ArgumentParser(description="Visualize a CoT-RAVEN .npz file with rules.")
    # parser.add_argument("npz_file", help="Path to the .npz file")
    # args = parser.parse_args()

    NPZ_FILE_PATH = "/home/scxhc1/nvme_data/cot_test/v2_test1/up_center_single_down_center_single/RAVEN_4542_train.npz"
    XML_FILE_PATH = os.path.splitext(NPZ_FILE_PATH)[0] + ".xml"

    # --------------------

    # 加载数据
    try:
        data = np.load(NPZ_FILE_PATH)
        img = data['image']
        target = int(data['target'])
    except FileNotFoundError:
        print(f"wrong path -> {NPZ_FILE_PATH}")
        exit()
    except KeyError:
        print(f"{NPZ_FILE_PATH} missing 'image' or 'target' data")
        exit()

    # 解析规则
    rules_info = parse_rules_from_xml(XML_FILE_PATH)
    rules_text = build_rules_text(rules_info, n_cols=5)

    # ====== 图形与网格布局（左：题面；右：规则栏） ======
    # 适当加宽，右侧留出竖栏
    fig = plt.figure(figsize=(11.8, 7.5))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.06)

    # 外层：1行2列 -> 左边主内容（上下两块），右边规则竖栏
    outer = gridspec.GridSpec(
        nrows=1, ncols=2,
        width_ratios=[6.0, 2.0],  # 调整右侧栏宽度（越大越宽）
        wspace=0.15
    )

    # 左侧再切两行：上(3×5 上下文) / 下(2×4 答案)
    left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0],
        height_ratios=(3, 2), hspace=0.22
    )

    # --- 1) 3×5 上下文网格 ---
    n_rows, n_cols = 3, 5
    context_spec = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=left[0],
        wspace=0.08, hspace=0.08
    )

    num_context = (n_rows * n_cols) - 1
    for i in range(n_rows * n_cols):
        ax = plt.Subplot(fig, context_spec[i])
        ax.set_xticks([])
        ax.set_yticks([])

        if i < (n_rows * n_cols - 1):  # 最后一个格子是问号
            ax.imshow(img[i, :, :], cmap='gray')
        else:
            ax.text(0.5, 0.5, '?', fontsize=40, ha='center', va='center', color='red')
            ax.set_frame_on(True)

        # 不再在单元格上方写规则标题了（移除 ax.set_title(...)）
        fig.add_subplot(ax)

    # --- 2) 2×4 答案网格 ---
    answer_spec = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=left[1],
        wspace=0.08, hspace=0.08
    )

    num_answers = 8
    for i in range(num_answers):
        if (num_context + i) >= len(img):
            break  # 防止候选答案少于8个时出错

        ax = plt.Subplot(fig, answer_spec[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img[num_context + i, :, :], cmap='gray')

        # 正确答案高亮
        if i == target:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)

        fig.add_subplot(ax)

    # --- 3) 右侧规则竖栏 ---
    ax_rules = plt.Subplot(fig, outer[1])
    ax_rules.set_xticks([])
    ax_rules.set_yticks([])
    ax_rules.axis('off')

    # 给规则栏加一个浅色边框/底
    # 你也可以去掉 bbox，按需美化
    ax_rules.text(
        0.02, 0.98, rules_text,
        va='top', ha='left',
        fontsize=8,
        family='monospace',
        linespacing=1.15,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", lw=1.0)
    )
    fig.add_subplot(ax_rules)

    # 标题与保存
    problem_name = Path(NPZ_FILE_PATH).stem
    distribution = Path(NPZ_FILE_PATH).parts[-2:-1]
    fig.suptitle(f"Problem: {problem_name} (Target: {target + 1})", fontsize=14)

    save_name = "./visual/" + distribution[0] + problem_name[5:] + ".pdf"
    fig.savefig(save_name)
    print(f"可视化图像已保存到: {save_name}")


if __name__ == "__main__":
    main()
