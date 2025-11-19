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
                # column_rule_groups = sample_rules(num_components)
                column_rule_groups = None
                while True:
                    candidate_rules = sample_rules(num_components)
                    if root.prune(candidate_rules) is not None:
                        column_rule_groups = candidate_rules
                        break
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

            # # --- 步骤 4: 生成干扰项 (I-RAVEN version)---
            # rules_for_last_step = all_column_rules[-1]
            # modifiable_attr = sample_attr_avail(rules_for_last_step, answer_AoT)
            # candidates = [answer_AoT]

            # attr_num = 3
            # if attr_num <= len(modifiable_attr):
            #     idx = np.random.choice(len(modifiable_attr), attr_num, replace=False)
            #     selected_attr = [modifiable_attr[i] for i in idx]
            # else:
            #     selected_attr = modifiable_attr

            # mode = None
            # pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Number']
            # if pos:
            #     pos = pos[0]
            #     selected_attr[pos], selected_attr[-1] = selected_attr[-1], selected_attr[pos]

            #     pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Position']
            #     if pos:
            #         mode = 'Position-Number'
            # values = []
            # if len(selected_attr) >= 3:
            #     mode_3 = None
            #     if mode == 'Position-Number':
            #         mode_3 = '3-Position-Number'
            #     for i in range(attr_num):
            #         component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[i][0], selected_attr[i][1], \
            #             selected_attr[i][3], selected_attr[i][4], \
            #             selected_attr[i][5]
            #         value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni,
            #                                             mode_3)
            #         values.append(value)
            #         tmp = []
            #         for j in candidates:
            #             new_AoT = copy.deepcopy(j)
            #             new_AoT.apply_new_value(component_idx, attr_name, value)
            #             tmp.append(new_AoT)
            #         candidates += tmp

            # elif len(selected_attr) == 2:
            #     component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
            #         selected_attr[0][3], selected_attr[0][4], \
            #         selected_attr[0][5]
            #     value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
            #     values.append(value)
            #     new_AoT = copy.deepcopy(answer_AoT)
            #     new_AoT.apply_new_value(component_idx, attr_name, value)
            #     candidates.append(new_AoT)
            #     component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[1][0], selected_attr[1][1], \
            #         selected_attr[1][3], selected_attr[1][4], \
            #         selected_attr[1][5]
            #     if mode == 'Position-Number':
            #         ran, qu = 6, 1
            #     else:
            #         ran, qu = 3, 2
            #     for i in range(ran):
            #         value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
            #         values.append(value)
            #         for j in range(qu):
            #             new_AoT = copy.deepcopy(candidates[j])
            #             new_AoT.apply_new_value(component_idx, attr_name, value)
            #             candidates.append(new_AoT)

            # elif len(selected_attr) == 1:
            #     component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
            #         selected_attr[0][3], selected_attr[0][4], \
            #         selected_attr[0][5]
            #     for i in range(7):
            #         value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
            #         values.append(value)
            #         new_AoT = copy.deepcopy(answer_AoT)
            #         new_AoT.apply_new_value(component_idx, attr_name, value)
            #         candidates.append(new_AoT)

            # random.shuffle(candidates)
            # answers = []
            # for candidate in candidates:
            #     answers.append(render_panel(candidate))

            # --- 步骤 4: 生成干扰项 (Hybrid: I-RAVEN Balanced + CoT Traps) ---
            rules_for_last_step = all_column_rules[-1]
            target_panel = answer_AoT
            
            # 准备候选集，先放入正确答案
            candidates = [target_panel]

            # ==========================================
            # Part A: CoT 时间陷阱 (基于 P13/P14) - 3个
            # ==========================================
            # 目的：生成看起来像“上一步”或“上上步”的选项，强迫模型进行完整推理
            p14_panel = all_panels[n_rows - 1][n_columns - 2]
            p13_panel = all_panels[n_rows - 1][n_columns - 3]
            
            # 定义陷阱来源和数量
            trap_sources = [
                (p14_panel, 2), # 2个来自 P14 (t-1)
                (p13_panel, 1)  # 1个来自 P13 (t-2)
            ]

            # 辅助函数：检查重复 (基于渲染结果)
            def is_duplicate(aot1, aot2):
                return np.array_equal(render_panel(aot1), render_panel(aot2))

            for src_panel, count in trap_sources:
                modifiable_attr = sample_attr_avail(rules_for_last_step, src_panel)
                if not modifiable_attr: # 兜底
                    src_panel = target_panel
                    modifiable_attr = sample_attr_avail(rules_for_last_step, target_panel)
                
                for _ in range(count):
                    # 尝试生成非重复的陷阱
                    for _ in range(10):
                        cand = copy.deepcopy(src_panel)
                        # 逻辑陷阱策略：优先保持原样(模拟没走完推理)，如果重复则微调1个属性
                        if is_duplicate(cand, target_panel):
                            # 如果 P14 == Target (例如Constant规则)，则必须修改
                            if modifiable_attr:
                                idx = np.random.choice(len(modifiable_attr))
                                c_idx, a_name, _, min_l, max_l, a_uni = modifiable_attr[idx]
                                val = cand.sample_new_value(c_idx, a_name, min_l, max_l, a_uni, None)
                                cand.apply_new_value(c_idx, a_name, val)
                        else:
                            # 如果 P14 != Target，有 50% 概率微调，50% 概率保持原样作为强干扰
                            if modifiable_attr and np.random.random() < 0.5:
                                idx = np.random.choice(len(modifiable_attr))
                                c_idx, a_name, _, min_l, max_l, a_uni = modifiable_attr[idx]
                                val = cand.sample_new_value(c_idx, a_name, min_l, max_l, a_uni, None)
                                cand.apply_new_value(c_idx, a_name, val)
                        
                        # 查重
                        is_unique = not is_duplicate(cand, target_panel)
                        if is_unique:
                            for exist in candidates:
                                if is_duplicate(cand, exist):
                                    is_unique = False; break
                        
                        if is_unique:
                            candidates.append(cand)
                            break

            # ==========================================
            # Part B: I-RAVEN 平衡干扰项 (基于 Target) - 补足 8 个
            # ==========================================
            # 目的：使用原版算法生成多属性修改的干扰项，保证属性分布的平衡性
            # 计算还需要生成多少个 (通常是 8 - 1 - 3 = 4 个)
            needed = 8 - len(candidates)
            
            if needed > 0:
                # 获取 Target 的可修改属性
                modifiable_attr = sample_attr_avail(rules_for_last_step, target_panel)
                
                # --- 复用 I-RAVEN 原版生成逻辑 (核心) ---
                # 这段逻辑保证了修改属性的组合性 (1个, 2个, 3个属性组合修改)
                
                # 1. 确定要修改的属性集合
                attr_num = 3
                if attr_num <= len(modifiable_attr):
                    idx = np.random.choice(len(modifiable_attr), attr_num, replace=False)
                    selected_attr = [modifiable_attr[i] for i in idx]
                else:
                    selected_attr = modifiable_attr

                # 2. 处理 Position-Number 联动
                mode = None
                pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Number']
                if pos:
                    pos = pos[0]
                    selected_attr[pos], selected_attr[-1] = selected_attr[-1], selected_attr[pos]
                    pos_chk = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Position']
                    if pos_chk: mode = 'Position-Number'

                # 3. 递归/循环生成
                # I-RAVEN 原逻辑是生成一大堆然后选，这里我们简化为生成直到填满
                while len(candidates) < 8:
                    cand = copy.deepcopy(target_panel)
                    
                    # 随机决定修改几个属性 (1~len)，模仿 ABT 的层级
                    num_to_mod = np.random.choice(range(1, len(selected_attr) + 1))
                    
                    # 按照 I-RAVEN 的方式修改属性
                    # 这里简化了原版复杂的 if-else 树，直接进行组合修改
                    current_attrs = selected_attr[:num_to_mod]
                    
                    for i in range(len(current_attrs)):
                        c_idx, a_name, _, min_l, max_l, a_uni = current_attrs[i][0], current_attrs[i][1], \
                            current_attrs[i][3], current_attrs[i][4], current_attrs[i][5]
                        
                        mode_param = mode if (mode == 'Position-Number' and i == len(current_attrs)-1) else None
                        if mode_param == 'Position-Number' and i < 2: mode_param = '3-Position-Number' # 简化的逻辑

                        val = cand.sample_new_value(c_idx, a_name, min_l, max_l, a_uni, mode_param)
                        cand.apply_new_value(c_idx, a_name, val)
                    
                    # 查重
                    is_unique = not is_duplicate(cand, target_panel)
                    if is_unique:
                        for exist in candidates:
                            if is_duplicate(cand, exist):
                                is_unique = False; break
                    
                    if is_unique:
                        candidates.append(cand)

            random.shuffle(candidates)
            
            # 最后 Prune 检查
            if len(candidates) < 8:
                # print(f"Skipping defective sample {k}: only {len(candidates)} candidates.")
                continue

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