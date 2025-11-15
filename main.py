# -*- coding: utf-8 -*-

import argparse
import copy
import os
import random

import numpy as np
from tqdm import trange

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single,
                        merge_component)  # <-- 修复：从 build_tree 导入
from const import IMAGE_SIZE
from rendering import render_panel
from sampling import sample_attr_avail, sample_rules
from serialize import dom_problem, serialize_aot, serialize_rules
from solver import solve


# --- 修复：删除这里的 merge_component 定义 ---
# def merge_component(dst_aot, src_aot, component_idx):
# ...
# --- 修复结束 ---


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    n_rows = 3
    n_columns = 5  # CoT-RAVEN.md中的 3xN
    r_base = 2  # 基础列的数量 (t=0, t=1)，为 2-arity 规则提供输入

    for key in list(all_configs.keys()):
        acc = 0
        for k in trange(args.num_samples):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            # root 是一个抽象的模板 (is_pg=False)
            root = all_configs[key]

            # 从抽象 root 确定组件数量
            num_components = len(root.children[0].children)

            # (3x5) 的面板网格
            all_panels = [[None for _ in range(n_columns)] for _ in range(n_rows)]
            all_column_rules = []  # 存储 t=2, 3, 4 列的规则

            # --- 步骤 1: 生成基础列 (t=0, t=1) ---
            for r in range(n_rows):
                for t in range(r_base):
                    all_panels[r][t] = root.sample()

            # --- 步骤 2: 生成递推列 (t=2, 3, 4) ---
            for t in range(r_base, n_columns):
                # 传入正确的 num_components
                column_rule_groups = sample_rules(num_components)
                all_column_rules.append(column_rule_groups)

                for r in range(n_rows):
                    previous_panels_in_row = all_panels[r][:t]
                    final_panel_for_row_col = None

                    for l in range(num_components):
                        rule_group_for_comp = column_rule_groups[l]
                        panel_template = copy.deepcopy(previous_panels_in_row[-1])
                        panel_in_progress = None

                        for i in range(len(rule_group_for_comp)):
                            rule = rule_group_for_comp[i]

                            arity = 1
                            if rule.name in ["Arithmetic", "Distribute_Three"]:
                                arity = 2

                            input_panels = previous_panels_in_row[-arity:]
                            in_aot = panel_template if i == 0 else panel_in_progress
                            panel_in_progress = rule.apply_rule(input_panels, in_aot=in_aot)

                        if l == 0:
                            final_panel_for_row_col = panel_in_progress
                        else:
                            merge_component(final_panel_for_row_col, panel_in_progress, l)

                    all_panels[r][t] = final_panel_for_row_col

            # --- 步骤 3: 准备上下文、答案和候选 ---
            answer_AoT = all_panels[n_rows - 1][n_columns - 1]
            context_list_flat = [p for row in all_panels for p in row]
            answer_index = (n_rows * n_columns) - 1
            context_list_flat[answer_index] = None
            imgs = [render_panel(p) if p is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8) for p in
                    context_list_flat]
            full_context_aot = [p for p in context_list_flat if p is not None]

            # --- 步骤 4: 生成干扰项 ---
            rules_for_last_step = all_column_rules[-1]
            modifiable_attr = sample_attr_avail(rules_for_last_step, answer_AoT)
            candidates = [answer_AoT]

            attr_num = 3
            if attr_num <= len(modifiable_attr):
                idx = np.random.choice(len(modifiable_attr), attr_num, replace=False)
                selected_attr = [modifiable_attr[i] for i in idx]
            else:
                selected_attr = modifiable_attr

            mode = None
            pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Number']
            if pos:
                pos = pos[0]
                selected_attr[pos], selected_attr[-1] = selected_attr[-1], selected_attr[pos]

                pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Position']
                if pos:
                    mode = 'Position-Number'
            values = []
            if len(selected_attr) >= 3:
                mode_3 = None
                if mode == 'Position-Number':
                    mode_3 = '3-Position-Number'
                for i in range(attr_num):
                    component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[i][0], selected_attr[i][1], \
                        selected_attr[i][3], selected_attr[i][4], \
                        selected_attr[i][5]
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni,
                                                        mode_3)
                    values.append(value)
                    tmp = []
                    for j in candidates:
                        new_AoT = copy.deepcopy(j)
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        tmp.append(new_AoT)
                    candidates += tmp

            elif len(selected_attr) == 2:
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
                    selected_attr[0][3], selected_attr[0][4], \
                    selected_attr[0][5]
                value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                values.append(value)
                new_AoT = copy.deepcopy(answer_AoT)
                new_AoT.apply_new_value(component_idx, attr_name, value)
                candidates.append(new_AoT)
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[1][0], selected_attr[1][1], \
                    selected_attr[1][3], selected_attr[1][4], \
                    selected_attr[1][5]
                if mode == 'Position-Number':
                    ran, qu = 6, 1
                else:
                    ran, qu = 3, 2
                for i in range(ran):
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                    values.append(value)
                    for j in range(qu):
                        new_AoT = copy.deepcopy(candidates[j])
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        candidates.append(new_AoT)

            elif len(selected_attr) == 1:
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
                    selected_attr[0][3], selected_attr[0][4], \
                    selected_attr[0][5]
                for i in range(7):
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                    values.append(value)
                    new_AoT = copy.deepcopy(answer_AoT)
                    new_AoT.apply_new_value(component_idx, attr_name, value)
                    candidates.append(new_AoT)

            random.shuffle(candidates)
            answers = []
            for candidate in candidates:
                answers.append(render_panel(candidate))

            # --- 步骤 5: 求解 ---
            context_panels_for_solver = all_panels[n_rows - 1][n_columns - r_base: n_columns - 1]
            image = imgs[:-1] + answers
            target = candidates.index(answer_AoT)

            predicted = solve(rules_for_last_step, context_panels_for_solver, candidates)

            # --- 步骤 6: 序列化 ---
            meta_matrix, meta_target = serialize_rules(rules_for_last_step)
            structure, meta_structure = serialize_aot(all_panels[0][0])

            np.savez("{}/{}/RAVEN_{}_{}.npz".format(args.save_dir, key, k, set_name), image=image,
                     target=target,
                     predict=predicted,
                     meta_matrix=meta_matrix,
                     meta_target=meta_target,
                     structure=structure,
                     meta_structure=meta_structure)
            with open("{}/{}/RAVEN_{}_{}.xml".format(args.save_dir, key, k, set_name), "wb") as f:
                dom = dom_problem(full_context_aot + candidates, all_column_rules)
                f.write(dom)

            if target == predicted:
                acc += 1
        print(("Accuracy of {}: {}".format(key, float(acc) / args.num_samples)))


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for I-RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=100,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="dataset",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=1234,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--fuse", type=int, default=0,
                                 help="whether to fuse different configurations")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")
    args = main_arg_parser.parse_args()

    all_configs = {
                    "center_single": build_center_single(),
                   "distribute_four": build_distribute_four(),
                   "distribute_nine": build_distribute_nine(),
                   "left_center_single_right_center_single": build_left_center_single_right_center_single(),
                   "up_center_single_down_center_single": build_up_center_single_down_center_single(),
                   "in_center_single_out_center_single": build_in_center_single_out_center_single(),
                   "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()
    }

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not args.fuse:
        for key in list(all_configs.keys()):
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        separate(args, all_configs)


if __name__ == "__main__":
    main()