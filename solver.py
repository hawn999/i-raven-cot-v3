# -*- coding: utf-8 -*-


import numpy as np
import copy


# 注意：这个求解器不再需要导入 main 或 rendering
# 它是一个纯粹的逻辑检查器


def solve(rule_groups, context, candidates):
    """
    Search-based Heuristic Solver (Corrected Reverse-Check Strategy).

    Arguments:
        rule_groups(list of list of Rule): 最后一列 (t=n) 的规则
        context(list of AoTNode): 最后一步所需的上下文 [panel_t-2, panel_t-1]
        candidates(list of AoTNode): 8个候选答案
    Returns:
        ans(int): index of the correct answer in the candidates
    """
    satisfied = [0] * len(candidates)

    if len(context) < 2:
        # 上下文不足，无法进行 2-arity 检查，只能随机猜
        return np.random.choice(len(candidates))

    for i, candidate in enumerate(candidates):
        score = 0
        # 遍历每个组件 (例如 'Left' 和 'Right')
        for component_idx in range(len(rule_groups)):
            rule_group = rule_groups[component_idx]

            # 1. 检查 Number/Position 规则
            rule_num_pos = rule_group[0]
            score += check_num_pos(rule_num_pos, context, candidate)

            # 2. 检查实体规则 (Type, Size, Color)
            for entity_rule in rule_group[1:]:
                score += check_entity(entity_rule, context, candidate)

        satisfied[i] = score

    satisfied = np.array(satisfied)
    # 找到最高分
    max_score = np.max(satisfied)

    # 检查是否有规则被应用（score > 0）。如果没有，随机猜测。
    # 并且，检查最高分是否*唯一*。如果多个候选得到满分（理论上不应发生），随机选一个。
    if max_score == 0:
        return np.random.choice(len(candidates))

    # 找到所有获得最高分的候选
    answer_set = np.where(satisfied == max_score)[0]

    # 从最高分中随机选一个（通常只有一个）
    return np.random.choice(answer_set)


def get_layouts(rule, context, candidate):
    """辅助函数：安全地提取 t-2, t-1 和 candidate 的布局"""
    component_idx = rule.component_idx
    try:
        layout_t_minus_2 = context[0].children[0].children[component_idx].children[0]
        layout_t_minus_1 = context[1].children[0].children[component_idx].children[0]
        layout_cand = candidate.children[0].children[component_idx].children[0]
        return layout_t_minus_2, layout_t_minus_1, layout_cand
    except IndexError:
        return None, None, None  # 缺少组件/布局


def check_num_pos(rule, context, candidate):
    """
    检查 Number/Position 规则。
    context = [Panel_t-2, Panel_t-1]
    """
    layout_t_minus_2, layout_t_minus_1, layout_cand = get_layouts(rule, context, candidate)
    if layout_cand is None:
        return 0  # 无法检查

    attr = rule.attr

    # --- 规则: Constant ---
    if rule.name == "Constant":
        num_t_minus_1 = layout_t_minus_1.number.get_value()
        num_cand = layout_cand.number.get_value()
        pos_t_minus_1 = set(layout_t_minus_1.position.get_value_idx())
        pos_cand = set(layout_cand.position.get_value_idx())

        # 必须 Number 和 Position 都保持不变
        if num_t_minus_1 == num_cand and pos_t_minus_1 == pos_cand:
            return 1

    # --- 规则: Progression ---
    elif rule.name == "Progression":
        if attr == "Number":
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            if (v2 - v1) == (v3 - v2) and (v2 - v1) == rule.value:
                return 1
        else:  # Position
            if not (
                    layout_t_minus_2.number.get_value() == layout_t_minus_1.number.get_value() == layout_cand.number.get_value()):
                return 0
            if layout_cand.number.get_value() == 0:
                return 1  # 空面板的 Progression 规则自动满足

            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())

            most_num = len(layout_cand.position.values)
            diff = rule.value

            expected_v2_pos = set((p + diff) % most_num for p in v1_pos)
            expected_v3_pos = set((p + diff) % most_num for p in v2_pos)

            if v2_pos == expected_v2_pos and v3_pos == expected_v3_pos:
                return 1

    # --- 规则: Arithmetic ---
    elif rule.name == "Arithmetic":
        if attr == "Number":
            # BUG 修复：Arithmetic(Number) 作用于 level (索引)，而不是 value (值)
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            if rule.value > 0:  # 加法
                if v3 == v1 + v2 + 1: return 1  # 匹配 Rule.py 的 (L+L+1)
            else:  # 减法
                if v3 == abs(v1 - v2): return 1  # 匹配 Rule.py 的 abs(L-L)
        else:  # Position
            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())
            if rule.value > 0:  # Union
                if v3_pos == (v1_pos | v2_pos): return 1
            else:  # Difference
                if v3_pos == (v1_pos - v2_pos): return 1

    # --- 规则: Distribute_Three ---
    elif rule.name == "Distribute_Three":
        if attr == "Number":
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            if v1 != v2 and v1 != v3 and v2 != v3:
                return 1
        else:  # Position
            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())
            if v1_pos != v2_pos and v1_pos != v3_pos and v2_pos != v3_pos:
                return 1

    return 0  # 规则不匹配


def check_entity(rule, context, candidate):
    """
    检查实体属性规则 (Type, Size, Color)。
    """
    layout_t_minus_2, layout_t_minus_1, layout_cand = get_layouts(rule, context, candidate)
    if layout_cand is None:
        return 0  # 无法检查

    attr = rule.attr

    # 实体规则要求面板非空，且属性一致
    is_consistent_v1 = check_consistency(layout_t_minus_2, attr)
    is_consistent_v2 = check_consistency(layout_t_minus_1, attr)
    is_consistent_v3 = check_consistency(layout_cand, attr)

    is_empty_v1 = layout_t_minus_2.number.get_value() == 0
    is_empty_v2 = layout_t_minus_1.number.get_value() == 0
    is_empty_v3 = layout_cand.number.get_value() == 0

    # 如果三个面板都一致（或为空），则继续检查
    if (is_consistent_v1 or is_empty_v1) and \
            (is_consistent_v2 or is_empty_v2) and \
            (is_consistent_v3 or is_empty_v3):

        # 如果三个都为空，规则满足
        if is_empty_v1 and is_empty_v2 and is_empty_v3:
            return 1

        # v3 为空，但 v1/v2 非空，这可能是一个合法的 Arithmetic(sub)
        if is_empty_v3 and (not is_empty_v1 or not is_empty_v2):
            if rule.name == "Arithmetic" and rule.value < 0:
                # 检查是否 v1 == v2 (例如, 5-5=0)
                if not is_empty_v1 and not is_empty_v2:
                    v1_val = getattr(layout_t_minus_2.children[0], attr.lower()).get_value_level()
                    v2_val = getattr(layout_t_minus_1.children[0], attr.lower()).get_value_level()
                    if v1_val == v2_val:
                        return 1
            return 0  # 空面板不满足其他规则

        # v1 或 v2 为空，但 v3 非空
        if (is_empty_v1 or is_empty_v2) and not is_empty_v3:
            if rule.name == "Arithmetic" and rule.value > 0:  # 0 + v2 = v3?
                v1 = getattr(layout_t_minus_2.children[0], attr.lower()).get_value_level() if not is_empty_v1 else 0
                v2 = getattr(layout_t_minus_1.children[0], attr.lower()).get_value_level() if not is_empty_v2 else 0
                v3 = getattr(layout_cand.children[0], attr.lower()).get_value_level()

                expected_v3 = (v1 + v2) if rule.attr == "Color" else (v1 + v2 + 1)
                if v3 == expected_v3: return 1
            return 0  # 空面板不满足其他规则

        # --- 常规情况：三个面板都非空 ---
        attr_lower = rule.attr.lower()
        v1 = getattr(layout_t_minus_2.children[0], attr_lower).get_value_level()
        v2 = getattr(layout_t_minus_1.children[0], attr_lower).get_value_level()
        v3 = getattr(layout_cand.children[0], attr_lower).get_value_level()

        # --- 规则: Constant ---
        if rule.name == "Constant":
            if v3 == v2:
                return 1

        # --- 规则: Progression ---
        elif rule.name == "Progression":
            if (v2 - v1) == (v3 - v2) and (v2 - v1) == rule.value:
                return 1

        # --- 规则: Arithmetic ---
        elif rule.name == "Arithmetic":
            if rule.value > 0:  # 加法
                expected_v3 = (v1 + v2) if rule.attr == "Color" else (v1 + v2 + 1)
                if v3 == expected_v3: return 1
            else:  # 减法
                expected_v3_val = (v1 - v2) if rule.attr == "Color" else (v1 - v2 - 1)
                if v3 == abs(expected_v3_val): return 1  # 匹配 Rule.py 中的 abs()

        # --- 规则: Distribute_Three ---
        elif rule.name == "Distribute_Three":
            if v1 != v2 and v1 != v3 and v2 != v3:
                return 1

    return 0


def check_consistency(layout, attr):
    """检查一个布局内的所有实体是否在某个属性上值都相同"""
    if not layout.children:
        return True  # 空布局被认为是“一致的”

    attr_name = attr.lower()
    first_val = getattr(layout.children[0], attr_name).get_value_level()
    for entity in layout.children[1:]:
        if getattr(entity, attr_name).get_value_level() != first_val:
            return False
    return True