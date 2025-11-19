# -*- coding: utf-8 -*-


import copy

import numpy as np

from const import (COLOR_MAX, COLOR_MIN, NUM_MAX, NUM_MIN, SIZE_MAX,
                   SIZE_MIN)
from AoT import Entity


def Rule_Wrapper(name, attr, param, component_idx):
    if name == "Constant":
        ret = Constant(name, attr, param, component_idx)
    elif name == "Progression":
        ret = Progression(name, attr, param, component_idx)
    elif name == "Arithmetic":
        ret = Arithmetic(name, attr, param, component_idx)
    elif name == "Distribute_Three":
        ret = Distribute_Three(name, attr, param, component_idx)
    else:
        raise ValueError("Unsupported Rule")
    return ret


class Rule:
    """General API for a rule.
    Priority order: Rule on Number/Position always comes first
    """

    def __init__(self, name, attr, params, component_idx=0):
        """Instantiate a rule by its name, attribute, paramter list and the component it applies to.
        Each rule should be applied to all entities in a component.
        Arguments:
            name(str): pre-defined name of the rule
            attr(str): pre-defined name of the attribute
            params(list): a list of possible parameters for it to sample
            component_idx(int): the index of the component to apply the rule
        """
        self.name = name
        self.attr = attr
        self.params = params
        self.component_idx = component_idx
        self.value = 0
        self.sample()

    def sample(self):
        """Sample a parameter from the parameter list.
        """
        if self.params is not None:
            self.value = np.random.choice(self.params)
            self.value = int(self.value)

    def apply_rule(self, aot_list, in_aot=None):
        """Apply the rule to a component in the AoT.
        Arguments:
            aot_list(list of AoTNode): a list of AoTs for reference
            in_aot(AoTNode): an AoT to apply the rule
        Returns:
            second_aot(AoTNode): a modified AoT
        """
        # Root -> Structure -> Component -> Layout -> Entity
        pass


class Constant(Rule):
    """Unary operator (1-arity). Nothing changes.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Constant, self).__init__(name, attr, param, component_idx)

    def apply_rule(self, aot_list, in_aot=None):
        # 1-arity 规则，只看 aot_list[-1]
        aot = aot_list[-1]
        if in_aot is None:
            in_aot = aot
        return copy.deepcopy(in_aot)


class Progression(Rule):
    """Unary operator (1-arity). Attribute difference on two consequetive Panels remains the same.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Progression, self).__init__(name, attr, param, component_idx)
        # 标志位在CoT模式下不再需要，因为我们是无状态的
        # self.first_col = True

    def apply_rule(self, aot_list, in_aot=None):
        # 1-arity 规则，只看 aot_list[-1]
        aot = aot_list[-1]
        current_layout = aot.children[0].children[self.component_idx].children[0]
        if in_aot is None:
            in_aot = aot
        second_aot = copy.deepcopy(in_aot)
        second_layout = second_aot.children[0].children[self.component_idx].children[0]

        if not current_layout.children:  # 如果没有实体，直接返回
            return second_aot

        if self.attr == "Number":
            second_layout.number.set_value_level(second_layout.number.get_value_level() + self.value)
            second_layout.position.sample(second_layout.number.get_value())
            pos = second_layout.position.get_value()
            del second_layout.children[:]
            for i in range(len(pos)):
                entity = copy.deepcopy(current_layout.children[0])
                entity.name = str(i)
                entity.bbox = pos[i]
                if not current_layout.uniformity.get_value():
                    entity.resample()
                second_layout.insert(entity)
        elif self.attr == "Position":
            second_pos_idx = (second_layout.position.get_value_idx() + self.value) % len(second_layout.position.values)
            second_layout.position.set_value_idx(second_pos_idx)
            second_bbox = second_layout.position.get_value()
            for i in range(len(second_bbox)):
                second_layout.children[i].bbox = second_bbox[i]
        elif self.attr == "Type":
            old_value_level = current_layout.children[0].type.get_value_level()
            for entity in second_layout.children:
                entity.type.set_value_level(old_value_level + self.value)
        elif self.attr == "Size":
            old_value_level = current_layout.children[0].size.get_value_level()
            for entity in second_layout.children:
                entity.size.set_value_level(old_value_level + self.value)
        elif self.attr == "Color":
            old_value_level = current_layout.children[0].color.get_value_level()
            for entity in second_layout.children:
                entity.color.set_value_level(old_value_level + self.value)
        else:
            raise ValueError("Unsupported attriubute")
        return second_aot


class Arithmetic(Rule):
    """Binary operator (2-arity). Panel_t = Panel_{t-2} + Panel_{t-1}.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Arithmetic, self).__init__(name, attr, param, component_idx)
        # 状态在CoT模式下被移除
        # self.color_count = 0
        # self.color_white_alarm = False

    def apply_rule(self, aot_list, in_aot=None):
        # 2-arity 规则，需要 2 个输入面板
        if len(aot_list) < 2:
            return copy.deepcopy(aot_list[-1])  # 输入不足

        first_aot = aot_list[-2]
        second_aot = aot_list[-1]

        first_layout = first_aot.children[0].children[self.component_idx].children[0]
        second_layout = second_aot.children[0].children[self.component_idx].children[0]

        if in_aot is None:
            in_aot = second_aot  # 新面板基于 t-1 面板
        new_aot = copy.deepcopy(in_aot)
        new_layout = new_aot.children[0].children[self.component_idx].children[0]

        if self.attr == "Number":
            first_layout_number_level = first_layout.number.get_value_level()
            second_layout_number_level = second_layout.number.get_value_level()

            if self.value > 0:
                total = first_layout_number_level + second_layout_number_level + 1
            else:
                # 确保减法不为负
                total = abs(first_layout_number_level - second_layout_number_level)

            total = np.clip(total, NUM_MIN, NUM_MAX)
            new_layout.number.set_value_level(total)

            new_layout.position.sample(new_layout.number.get_value())
            pos = new_layout.position.get_value()
            del new_layout.children[:]
            for i in range(len(pos)):
                if second_layout.children:
                    entity = copy.deepcopy(second_layout.children[0])
                else:
                    entity = Entity(name=str(i), bbox=pos[i], entity_constraint=second_layout.entity_constraint)
                entity.name = str(i)
                entity.bbox = pos[i]
                if not second_layout.uniformity.get_value():
                    entity.resample()
                new_layout.insert(entity)

        elif self.attr == "Position":
            first_layout_value_idx = first_layout.position.get_value_idx()
            second_layout_value_idx = second_layout.position.get_value_idx()

            if self.value > 0:
                new_pos_idx = set(first_layout_value_idx) | set(second_layout_value_idx)
            else:
                new_pos_idx = set(first_layout_value_idx) - set(second_layout_value_idx)

            if not new_pos_idx:
                return None

            new_layout.number.set_value_level(len(new_pos_idx) - 1)
            new_layout.position.set_value_idx(np.array(list(new_pos_idx)))

            pos = new_layout.position.get_value()
            del new_layout.children[:]
            for i in range(len(pos)):
                if second_layout.children:
                    entity = copy.deepcopy(second_layout.children[0])
                else:
                    entity = Entity(name=str(i), bbox=pos[i], entity_constraint=second_layout.entity_constraint)
                entity.name = str(i)
                entity.bbox = pos[i]
                if not second_layout.uniformity.get_value():
                    entity.resample()
                new_layout.insert(entity)

        elif self.attr == "Size":
            if not first_layout.children or not second_layout.children:
                return new_aot
            first_layout_size_level = first_layout.children[0].size.get_value_level()
            second_layout_size_level = second_layout.children[0].size.get_value_level()

            if self.value > 0:
                new_size_value_level = first_layout_size_level + second_layout_size_level + 1
            else:
                new_size_value_level = abs(first_layout_size_level - second_layout_size_level - 1)

            new_size_value_level = np.clip(new_size_value_level, SIZE_MIN, SIZE_MAX)
            for entity in new_layout.children:
                entity.size.set_value_level(new_size_value_level)

        elif self.attr == "Color":
            if not first_layout.children or not second_layout.children:
                return new_aot
            first_layout_color_level = first_layout.children[0].color.get_value_level()
            second_layout_color_level = second_layout.children[0].color.get_value_level()

            if self.value > 0:
                new_color_value_level = first_layout_color_level + second_layout_color_level
            else:
                new_color_value_level = abs(first_layout_color_level - second_layout_color_level)

            new_color_value_level = np.clip(new_color_value_level, COLOR_MIN, COLOR_MAX)
            for entity in new_layout.children:
                entity.color.set_value_level(new_color_value_level)
        else:
            raise ValueError("Unsupported attriubute")
        return new_aot


class Distribute_Three(Rule):
    """Binary operator (2-arity). V_t = Distribute_Three(V_{t-1}, V_{t-2}).
    新逻辑：V_t (或 V3) 是从总值集中选择的、一个与 V_{t-1}(V2) 和 V_{t-2}(V1) *都不同*的值。
    """

    def __init__(self, name, attr, param, component_idx):
        super(Distribute_Three, self).__init__(name, attr, param, component_idx)
        # 移除所有状态 (self.value_levels, self.count)

    def apply_rule(self, aot_list, in_aot=None):
        # 这是一个 2-arity 规则，需要 2 个输入面板
        if len(aot_list) < 2:
            return copy.deepcopy(aot_list[-1])  # 输入不足

        first_aot = aot_list[-2]  # V1
        second_aot = aot_list[-1]  # V2

        first_layout = first_aot.children[0].children[self.component_idx].children[0]
        second_layout = second_aot.children[0].children[self.component_idx].children[0]

        if in_aot is None:
            in_aot = second_aot

        new_aot = copy.deepcopy(in_aot)
        new_layout = new_aot.children[0].children[self.component_idx].children[0]

        # --- 获取原始约束边界 ---
        if self.attr in ["Number", "Position"]:
            # Position 规则也受 Number 的约束
            constraint_key = "Number" if self.attr == "Number" else "Number"
            # 必须使用 orig_layout_constraint 才能拿到完整的值范围
            min_level_orig = new_layout.orig_layout_constraint[constraint_key][0]
            max_level_orig = new_layout.orig_layout_constraint[constraint_key][1]
        elif self.attr in ["Type", "Size", "Color"]:
            min_level_orig = new_layout.orig_entity_constraint[self.attr][0]
            max_level_orig = new_layout.orig_entity_constraint[self.attr][1]
        else:
            raise ValueError("Distribute_Three 不支持此属性: {}".format(self.attr))

        all_value_levels = list(range(min_level_orig, max_level_orig + 1))

        # 如果可选值少于3个，规则无法生效，返回原样拷贝
        if len(all_value_levels) < 3:
            return new_aot

        if self.attr == "Number":
            v1_level = first_layout.number.get_value_level()
            v2_level = second_layout.number.get_value_level()

            # 找到一个与 v1, v2 都不同的 v3
            available = set(all_value_levels) - {v1_level, v2_level}
            if not available:  # 如果 v1, v2 占满了（比如只有3个值），就从除了v1,v2之外的重选
                available = set(all_value_levels) - {v1_level, v2_level}
                if not available: available = set(all_value_levels)  # 如果只有2个或1个值，就从中选

            v3_level = np.random.choice(list(available))
            new_layout.number.set_value_level(v3_level)

            # 基于新 Number 重新采样 Position 和实体
            new_layout.position.sample(new_layout.number.get_value())
            pos = new_layout.position.get_value()
            del new_layout.children[:]
            for i in range(len(pos)):
                if second_layout.children:
                    entity = copy.deepcopy(second_layout.children[0])
                else:
                    entity = Entity(name=str(i), bbox=pos[i], entity_constraint=second_layout.entity_constraint)
                entity.name = str(i)
                entity.bbox = pos[i]
                if not second_layout.uniformity.get_value():
                    entity.resample()
                new_layout.insert(entity)

        elif self.attr == "Position":
            # Position 的 Distribute_Three 意味着3个不同的位置*集合*
            v1_idx_set = set(first_layout.position.get_value_idx())
            v2_idx_set = set(second_layout.position.get_value_idx())
            num = new_layout.number.get_value()  # 保持与 t-1 面板相同的实体数量

            # 循环直到找到一个不同的位置集
            attempts = 0
            while attempts < 10:  # 防止死循环
                v3_idx = new_layout.position.sample_new(num)
                v3_idx_set = set(v3_idx)
                if v3_idx_set != v1_idx_set and v3_idx_set != v2_idx_set:
                    break
                attempts += 1

            new_layout.position.set_value_idx(v3_idx)
            pos = new_layout.position.get_value()
            if len(pos) == len(new_layout.children):
                for i in range(len(pos)):
                    new_layout.children[i].bbox = pos[i]

        elif self.attr in ["Type", "Size", "Color"]:
            # 实体属性（Type, Size, Color）必须是 uniform 的
            if not first_layout.children or not second_layout.children:
                return new_aot  # 空面板无法应用

            attr_lower = self.attr.lower()
            v1_level = getattr(first_layout.children[0], attr_lower).get_value_level()
            v2_level = getattr(second_layout.children[0], attr_lower).get_value_level()

            available = set(all_value_levels) - {v1_level, v2_level}
            if not available:
                available = set(all_value_levels)

            v3_level = np.random.choice(list(available))

            # 将 v3 应用到新面板的所有实体上
            for entity in new_layout.children:
                getattr(entity, attr_lower).set_value_level(v3_level)

        return new_aot