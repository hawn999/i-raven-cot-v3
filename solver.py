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
        return 0  # 缺少组件/布局，无法检查

    attr = rule.attr

    # --- Constant: 仅检查被声明的属性 ---
    if rule.name == "Constant":
        if attr == "Number":
            return int(layout_t_minus_1.number.get_value() ==
                       layout_cand.number.get_value())
        elif attr == "Position":
            return int(set(layout_t_minus_1.position.get_value_idx()) ==
                       set(layout_cand.position.get_value_idx()))
        elif attr == "Number/Position":  # 仅少数组合规则使用
            return int((layout_t_minus_1.number.get_value() ==
                        layout_cand.number.get_value()) and
                       (set(layout_t_minus_1.position.get_value_idx()) ==
                        set(layout_cand.position.get_value_idx())))
        return 0

    # --- Progression ---
    elif rule.name == "Progression":
        if attr == "Number":
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            return int((v2 - v1) == (v3 - v2) == rule.value)
        else:  # Position: 循环移位
            # 先确保三个面板 Number 一致或满足位置可比
            if not (layout_t_minus_2.number.get_value() ==
                    layout_t_minus_1.number.get_value() ==
                    layout_cand.number.get_value()):
                return 0
            if layout_cand.number.get_value() == 0:
                return 1  # 三者皆空，视为满足
            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())
            most_num = len(layout_cand.position.values)
            diff = rule.value
            expected_v2_pos = set((p + diff) % most_num for p in v1_pos)
            expected_v3_pos = set((p + diff) % most_num for p in v2_pos)
            return int(v2_pos == expected_v2_pos and v3_pos == expected_v3_pos)

    # --- Arithmetic ---
    elif rule.name == "Arithmetic":
        if attr == "Number":
            # Number 的算术作用在 level 上；加法 +1 偏置，减法取绝对值
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            if rule.value > 0:   # 加
                return int(v3 == v1 + v2 + 1)
            else:                # 减
                return int(v3 == abs(v1 - v2))
        else:  # Position: 并/差
            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())
            if rule.value > 0:   # union
                return int(v3_pos == (v1_pos | v2_pos))
            else:                # diff
                return int(v3_pos == (v1_pos - v2_pos))

    # --- Distribute_Three ---
    elif rule.name == "Distribute_Three":
        if attr == "Number":
            v1 = layout_t_minus_2.number.get_value_level()
            v2 = layout_t_minus_1.number.get_value_level()
            v3 = layout_cand.number.get_value_level()
            return int(v1 != v2 and v1 != v3 and v2 != v3)
        else:
            v1_pos = set(layout_t_minus_2.position.get_value_idx())
            v2_pos = set(layout_t_minus_1.position.get_value_idx())
            v3_pos = set(layout_cand.position.get_value_idx())
            return int(v1_pos != v2_pos and v1_pos != v3_pos and v2_pos != v3_pos)

    return 0

def check_entity(rule, context, candidate):
    """
    检查实体属性规则 (Type, Size, Color)。
    - 空面板统一按 len(children)==0 判定
    - Number/Size 带 ±1 偏置；Color 无偏置
    """
    layout_t_minus_2, layout_t_minus_1, layout_cand = get_layouts(rule, context, candidate)
    if layout_cand is None:
        return 0

    attr = rule.attr
    attr_lower = attr.lower()

    def _is_empty(layout):
        return len(layout.children) == 0

    def _consistent(layout):
        """布局内该属性是否一致"""
        if not layout.children:
            return True
        v0 = getattr(layout.children[0], attr_lower).get_value_level()
        for ent in layout.children[1:]:
            if getattr(ent, attr_lower).get_value_level() != v0:
                return False
        return True

    is_empty_v1, is_empty_v2, is_empty_v3 = map(_is_empty, (layout_t_minus_2, layout_t_minus_1, layout_cand))
    is_consistent_v1, is_consistent_v2, is_consistent_v3 = map(_consistent, (layout_t_minus_2, layout_t_minus_1, layout_cand))

    # 三个都空：规则满足
    if is_empty_v1 and is_empty_v2 and is_empty_v3:
        return 1

    # 若有非空，要求一致性或空可跳过该列比较
    if not ((is_consistent_v1 or is_empty_v1) and
            (is_consistent_v2 or is_empty_v2) and
            (is_consistent_v3 or is_empty_v3)):
        return 0

    # 取 level 值（空则不取）
    if not is_empty_v1:
        v1 = getattr(layout_t_minus_2.children[0], attr_lower).get_value_level()
    if not is_empty_v2:
        v2 = getattr(layout_t_minus_1.children[0], attr_lower).get_value_level()
    if not is_empty_v3:
        v3 = getattr(layout_cand.children[0], attr_lower).get_value_level()

    # --- Constant ---
    if rule.name == "Constant":
        # 空 → 空 已在前面返回；现为至少 v2/v3 非空
        if not is_empty_v2 and not is_empty_v3:
            return int(v3 == v2)
        return 0

    # --- Progression ---
    elif rule.name == "Progression":
        if not (is_empty_v1 or is_empty_v2 or is_empty_v3):
            return int((v2 - v1) == (v3 - v2) == rule.value)
        return 0

    # --- Arithmetic ---
    elif rule.name == "Arithmetic":
        # 处理因空引起的边界情况
        if rule.value > 0:
            # 加法：Color 无偏置，其它 +1 偏置
            if is_empty_v1 and is_empty_v2 and not is_empty_v3:
                # 0+0 不产生明确信号，判 0
                return 0
            if not (is_empty_v1 or is_empty_v2 or is_empty_v3):
                expected_v3 = (v1 + v2) if attr == "Color" else (v1 + v2 + 1)
                return int(v3 == expected_v3)
        else:
            # 减法：Color 无偏置，其它 (v1 - v2 - 1) 的绝对值
            if not (is_empty_v1 or is_empty_v2) and is_empty_v3:
                # 例如 5-5 → 0（空）
                return 1
            if not (is_empty_v1 or is_empty_v2 or is_empty_v3):
                expected_v3 = (v1 - v2) if attr == "Color" else (v1 - v2 - 1)
                return int(v3 == abs(expected_v3))
        return 0

    # --- Distribute_Three ---
    elif rule.name == "Distribute_Three":
        if not (is_empty_v1 or is_empty_v2 or is_empty_v3):
            return int(v1 != v2 and v1 != v3 and v2 != v3)
        return 0

    return 0

def solve_with_scores(rule_groups, context, candidates):
    """
    返回每个候选的得分，并给出是否“有且仅有一个最高分”。
    用于生成阶段强制唯一解：max>0 且 top-1 唯一。
    """
    scores = []
    for cand in candidates:
        s = 0
        for component_idx, rule_group in enumerate(rule_groups):
            rule_num_pos = rule_group[0]
            s += check_num_pos(rule_num_pos, context, cand)
            for entity_rule in rule_group[1:]:
                s += check_entity(entity_rule, context, cand)
        scores.append(s)
    scores = np.array(scores)
    n_top = np.count_nonzero(scores == scores.max())
    ok = (scores.max() > 0) and (n_top == 1)
    return scores, ok

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