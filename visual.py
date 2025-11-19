import matplotlib
# -------------------------------------------------------------------
# 修复：添加这两行，解决 Qt/xcb 错误
# 必须在导入 pyplot 之前设置 'Agg' 后端。
matplotlib.use('Agg')
# -------------------------------------------------------------------

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import argparse
import os
import glob
import random
import xml.etree.ElementTree as ET

# 默认的配置（子目录）列表
DEFAULT_CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single"
]

def parse_rules_from_xml(xml_path):
    """
    从 .xml 文件中解析规则信息
    返回一个字典，键是列索引 (str)，值是该列的规则字符串列表
    """
    if not os.path.exists(xml_path):
        # print(f"Warning: XML file not found at {xml_path}")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error: Failed to parse XML {xml_path}")
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
    """
    if not rules_info:
        return "no rule in XML"

    # 将 key 转成 int 后排序，保证顺序稳定
    try:
        sorted_cols = sorted(int(k) for k in rules_info.keys())
    except Exception:
        sorted_cols = sorted(rules_info.keys())

    sections = []
    for col in sorted_cols:
        key = str(col)
        if key not in rules_info or len(rules_info[key]) == 0:
            continue
        header = f"Col {col}"
        lines = [f"- {s}" for s in rules_info[key]]
        sections.append(header + "\n" + "\n".join(lines))

    if not sections:
        return "no visible rules"

    return "\n\n".join(sections)


def visualize_npz(npz_file_path, save_dir):
    """
    为单个 .npz 文件生成可视化图像
    """
    xml_file_path = os.path.splitext(npz_file_path)[0] + ".xml"

    # --- 加载数据 ---
    try:
        data = np.load(npz_file_path)
        img = data['image']
        target = int(data['target'])
    except FileNotFoundError:
        print(f"Error: wrong path -> {npz_file_path}")
        return
    except KeyError:
        print(f"Error: {npz_file_path} missing 'image' or 'target' data")
        return
    except Exception as e:
        print(f"Error: loading {npz_file_path}: {e}")
        return

    # --- 解析规则 ---
    rules_info = parse_rules_from_xml(xml_file_path)
    if rules_info is None:
        rules_text = "XML not found"
    else:
        rules_text = build_rules_text(rules_info, n_cols=5)

    # ====== 图形与网格布局 ======
    fig = plt.figure(figsize=(11.8, 7.5))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.06)

    outer = gridspec.GridSpec(
        nrows=1, ncols=2,
        width_ratios=[6.0, 2.0],
        wspace=0.15
    )

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

        if i < (n_rows * n_cols - 1):
            if i < len(img):
                ax.imshow(img[i, :, :], cmap='gray')
            else:
                ax.text(0.5, 0.5, 'X', fontsize=20, ha='center', va='center', color='gray') # 数据缺失
        else:
            ax.text(0.5, 0.5, '?', fontsize=40, ha='center', va='center', color='red')
            ax.set_frame_on(True)
        fig.add_subplot(ax)

    # --- 2) 2×4 答案网格 ---
    answer_spec = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=left[1],
        wspace=0.08, hspace=0.08
    )

    num_answers = 8
    for i in range(num_answers):
        img_index = num_context + i
        if img_index >= len(img):
            break  # 防止候选答案少于8个

        ax = plt.Subplot(fig, answer_spec[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img[img_index, :, :], cmap='gray')

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

    # --- 标题与保存 ---
    problem_name = Path(npz_file_path).stem
    fig.suptitle(f"Problem: {problem_name} (Target: {target + 1})", fontsize=14)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用 .png 格式以加快速度
    save_name = os.path.join(save_dir, problem_name + ".png")
    
    try:
        fig.savefig(save_name)
    except Exception as e:
        print(f"Error saving {save_name}: {e}")
    
    plt.close(fig) # 关闭图像，释放内存


def main():
    parser = argparse.ArgumentParser(description="Visualize CoT-RAVEN .npz files with rules.")
    parser.add_argument("--dataset_dir", default="./dataset_test2",
                        help="Path to the main dataset directory (e.g., './dataset')")
    parser.add_argument("--save_dir", default="./visual/test2_10000",
                        help="Directory to save visualization images")
    parser.add_argument("--config", nargs='+', default=None, 
                        help=f"Specific configs (sub-dirs) to visualize (e.g., center_single distribute_four). Default: all found ({len(DEFAULT_CONFIGS)})")
    
    # --- 变化 1：修改帮助文本 ---
    parser.add_argument("--num_vis", type=int, default=100,
                        help="Number of samples to visualize *per config* (total)")
    
    parser.add_argument("--random_sample", action='store_true', default=True, 
                        help="Randomly sample 'num_vis' files. (Default: True)")
    parser.add_argument("--no_random_sample", action='store_false', dest='random_sample',
                        help="Select the first 'num_vis' files instead of random sampling.")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        visualize_npz(args.dataset_dir, args.save_dir)
        return

    # 确定要处理哪些配置
    if args.config:
        configs_to_process = args.config
    else:
        # 如果未指定，则查找所有默认配置
        configs_to_process = []
        for cfg in DEFAULT_CONFIGS:
            if os.path.isdir(os.path.join(args.dataset_dir, cfg)):
                configs_to_process.append(cfg)
        if not configs_to_process:
            print(f"No default config directories found in {args.dataset_dir}. Specify --config or check path.")
            return
            
    print(f"Starting visualization...")
    print(f"Source: {args.dataset_dir}")
    print(f"Output: {args.save_dir}")
    print(f"Configs: {', '.join(configs_to_process)}")
    # --- 变化 2：修改打印信息 ---
    print(f"Samples per config: {args.num_vis} (Random: {args.random_sample})")
    print("-" * 30)

    # 遍历每个配置 (例如 "center_single")
    for config_name in configs_to_process:
        config_dir = os.path.join(args.dataset_dir, config_name)
        if not os.path.isdir(config_dir):
            print(f"Skipping '{config_name}': directory not found.")
            continue

        print(f"Processing config: {config_name}")

        # 为该配置创建输出子目录
        output_config_dir = os.path.join(args.save_dir, config_name)
        os.makedirs(output_config_dir, exist_ok=True)

        pattern = os.path.join(config_dir, "RAVEN_*.npz")
        npz_files = glob.glob(pattern)
        
        if not npz_files:
            print(f"  No .npz files found for config: {config_name}")
            continue

        # 根据参数选择文件
        selected_files = []
        if args.num_vis > 0:
            if args.random_sample:
                k = min(args.num_vis, len(npz_files))
                selected_files = random.sample(npz_files, k)
            else:
                npz_files.sort() # 确保顺序
                k = min(args.num_vis, len(npz_files))
                selected_files = npz_files[:k]
        
        if not selected_files:
            continue

        # --- 变化 4：修改打印信息 ---
        print(f"  Found {len(npz_files)} total files. Visualizing {len(selected_files)} samples...")

        # 为选中的文件生成图像
        for npz_path in selected_files:
            # print(f"    -> {Path(npz_path).name}")
            visualize_npz(npz_path, output_config_dir)

    print("-" * 30)
    print("Visualization batch complete.")


if __name__ == "__main__":
    main()