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
    """Unary operator. Nothing changes.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Constant, self).__init__(name, attr, param, component_idx)

    def apply_rule(self, aot_list, in_aot=None):
        aot = aot_list[-1]
        if in_aot is None:
            in_aot = aot
        return copy.deepcopy(in_aot)


class Progression(Rule):
    """Unary operator. Attribute difference on two consequetive Panels remains the same.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Progression, self).__init__(name, attr, param, component_idx)
        # Flag to trigger consistency of the attribute in the first column
        self.first_col = True

    def apply_rule(self, aot_list, in_aot=None):
        aot = aot_list[-1]
        current_layout = aot.children[0].children[self.component_idx].children[0]
        if in_aot is None:
            in_aot = aot
        second_aot = copy.deepcopy(in_aot)
        second_layout = second_aot.children[0].children[self.component_idx].children[0]

        #
        #
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
            # enforce value consistency
            if self.first_col and not current_layout.uniformity.get_value():
                for entity in current_layout.children:
                    entity.type.set_value_level(old_value_level)
            for entity in second_layout.children:
                entity.type.set_value_level(old_value_level + self.value)
        elif self.attr == "Size":
            old_value_level = current_layout.children[0].size.get_value_level()
            # enforce value consistency
            if self.first_col and not current_layout.uniformity.get_value():
                for entity in current_layout.children:
                    entity.size.set_value_level(old_value_level)
            for entity in second_layout.children:
                entity.size.set_value_level(old_value_level + self.value)
        elif self.attr == "Color":
            old_value_level = current_layout.children[0].color.get_value_level()
            # enforce value consistency
            if self.first_col and not current_layout.uniformity.get_value():
                for entity in current_layout.children:
                    entity.color.set_value_level(old_value_level)
            for entity in second_layout.children:
                entity.color.set_value_level(old_value_level + self.value)
        else:
            raise ValueError("Unsupported attriubute")
        self.first_col = False  # No longer the first column
        return second_aot


class Arithmetic(Rule):
    """Binary operator. Basically: Panel_n = Panel_{n-2} + Panel_{n-1}.
    For Position: + means SET_UNION and - SET_DIFF.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Arithmetic, self).__init__(name, attr, param, component_idx)
        self.color_count = 0
        self.color_white_alarm = False

    def apply_rule(self, aot_list, in_aot=None):
        if len(aot_list) < 2:
            # Not enough panels to apply arithmetic rule, return a copy of the last one
            return copy.deepcopy(aot_list[-1])

        first_aot = aot_list[-2]
        second_aot = aot_list[-1]

        first_layout = first_aot.children[0].children[self.component_idx].children[0]
        second_layout = second_aot.children[0].children[self.component_idx].children[0]

        if in_aot is None:
            in_aot = second_aot  # The new panel is based on the last one

        new_aot = copy.deepcopy(in_aot)
        new_layout = new_aot.children[0].children[self.component_idx].children[0]

        if self.attr == "Number":
            first_layout_number_level = first_layout.number.get_value_level()
            second_layout_number_level = second_layout.number.get_value_level()

            if self.value > 0:
                total = first_layout_number_level + second_layout_number_level + 1
            else:
                total = first_layout_number_level - second_layout_number_level

            total = np.clip(total, NUM_MIN, NUM_MAX)
            new_layout.number.set_value_level(total)

            new_layout.position.sample(new_layout.number.get_value())
            pos = new_layout.position.get_value()
            del new_layout.children[:]
            for i in range(len(pos)):
                # Handle cases where there might be no entities to copy from
                if second_layout.children:
                    entity = copy.deepcopy(second_layout.children[0])
                else:
                    # If no entities, create a new one based on constraints
                    entity = Entity(name=str(i), bbox=pos[i], entity_constraint=second_layout.entity_constraint)
                entity.name = str(i)
                entity.bbox = pos[i]
                if not second_layout.uniformity.get_value():
                    entity.resample()
                new_layout.insert(entity)

        elif self.attr == "Position":
            # ADD is interpreted as SET_UNION; SUB is interpreted as SET_DIFF
            first_layout_value_idx = first_layout.position.get_value_idx()
            second_layout_value_idx = second_layout.position.get_value_idx()

            if self.value > 0:
                new_pos_idx = set(first_layout_value_idx) | set(second_layout_value_idx)
            else:
                new_pos_idx = set(first_layout_value_idx) - set(second_layout_value_idx)

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
            # Ensure there are entities to get size from
            if not first_layout.children or not second_layout.children:
                return new_aot

            first_layout_size_level = first_layout.children[0].size.get_value_level()
            second_layout_size_level = second_layout.children[0].size.get_value_level()

            if self.value > 0:
                new_size_value_level = first_layout_size_level + second_layout_size_level + 1
            else:
                new_size_value_level = first_layout_size_level - second_layout_size_level - 1

            new_size_value_level = np.clip(new_size_value_level, SIZE_MIN, SIZE_MAX)
            for entity in new_layout.children:
                entity.size.set_value_level(new_size_value_level)

        elif self.attr == "Color":
            self.color_count += 1
            if not first_layout.children or not second_layout.children:
                return new_aot
            first_layout_color_level = first_layout.children[0].color.get_value_level()
            second_layout_color_level = second_layout.children[0].color.get_value_level()

            if self.value > 0:
                new_color_value_level = first_layout_color_level + second_layout_color_level
            else:
                new_color_value_level = first_layout_color_level - second_layout_color_level

            new_color_value_level = np.clip(new_color_value_level, COLOR_MIN, COLOR_MAX)
            for entity in new_layout.children:
                entity.color.set_value_level(new_color_value_level)

        else:
            raise ValueError("Unsupported attriubute")

        return new_aot


class Distribute_Three(Rule):
    """Ternay operator. Three values across the columns form a fixed set.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Distribute_Three, self).__init__(name, attr, param, component_idx)
        self.value_levels = []
        self.count = 0

    def apply_rule(self, aot_list, in_aot=None):
        aot = aot_list[-1]
        current_layout = aot.children[0].children[self.component_idx].children[0]
        if in_aot is None:
            in_aot = aot
        second_aot = copy.deepcopy(in_aot)
        second_layout = second_aot.children[0].children[self.component_idx].children[0]
        if self.attr == "Number":
            if self.count == 0:
                all_value_levels = list(range(current_layout.layout_constraint["Number"][0],
                                              current_layout.layout_constraint["Number"][1] + 1))
                current_value_level = current_layout.number.get_value_level()
                idx = all_value_levels.index(current_value_level)
                all_value_levels.pop(idx)
                three_value_levels = np.random.choice(all_value_levels, 2, False)
                three_value_levels = np.insert(three_value_levels, 0, current_value_level)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                second_layout.number.set_value_level(self.value_levels[0][1])
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    current_layout.number.set_value_level(self.value_levels[row][0])
                    current_layout.resample()
                    second_aot = copy.deepcopy(aot)
                    second_layout = second_aot.children[0].children[self.component_idx].children[0]
                    second_layout.number.set_value_level(self.value_levels[row][1])
                else:
                    second_layout.number.set_value_level(self.value_levels[row][2])
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
            self.count = (self.count + 1) % 6
        elif self.attr == "Position":
            if self.count == 0:
                # sample new does not change value_level/value_idx
                num = current_layout.number.get_value()
                pos_0 = current_layout.position.get_value_idx()
                pos_1 = current_layout.position.sample_new(num)
                pos_2 = current_layout.position.sample_new(num, [pos_1])
                three_value_levels = np.array([pos_0, pos_1, pos_2])
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                second_layout.position.set_value_idx(self.value_levels[0][1])
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    current_layout.number.set_value_level(len(self.value_levels[row][0]) - 1)
                    current_layout.resample()
                    current_layout.position.set_value_idx(self.value_levels[row][0])
                    pos = current_layout.position.get_value()
                    for i in range(len(pos)):
                        entity = current_layout.children[i]
                        entity.bbox = pos[i]
                    second_aot = copy.deepcopy(aot)
                    second_layout = second_aot.children[0].children[self.component_idx].children[0]
                    second_layout.position.set_value_idx(self.value_levels[row][1])
                else:
                    second_layout.position.set_value_idx(self.value_levels[row][2])
            pos = second_layout.position.get_value()
            for i in range(len(pos)):
                entity = second_layout.children[i]
                entity.bbox = pos[i]
            self.count = (self.count + 1) % 6
        elif self.attr == "Type":
            if self.count == 0:
                all_value_levels = list(range(current_layout.entity_constraint["Type"][0],
                                              current_layout.entity_constraint["Type"][1] + 1))
                # if np.random.uniform() >= 0.5 and 0 not in all_value_levels:
                #     all_value_levels.insert(0, 0)
                three_value_levels = np.random.choice(all_value_levels, 3, False)
                np.random.shuffle(three_value_levels)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                for entity in current_layout.children:
                    entity.type.set_value_level(self.value_levels[0][0])
                for entity in second_layout.children:
                    entity.type.set_value_level(self.value_levels[0][1])
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    value_level = self.value_levels[row][0]
                    for entity in current_layout.children:
                        entity.type.set_value_level(value_level)
                    value_level = self.value_levels[row][1]
                    for entity in second_layout.children:
                        entity.type.set_value_level(value_level)
                else:
                    value_level = self.value_levels[row][2]
                    for entity in second_layout.children:
                        entity.type.set_value_level(value_level)
            self.count = (self.count + 1) % 6
        elif self.attr == "Size":
            if self.count == 0:
                all_value_levels = list(range(current_layout.entity_constraint["Size"][0],
                                              current_layout.entity_constraint["Size"][1] + 1))
                three_value_levels = np.random.choice(all_value_levels, 3, False)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                for entity in current_layout.children:
                    entity.size.set_value_level(self.value_levels[0][0])
                for entity in second_layout.children:
                    entity.size.set_value_level(self.value_levels[0][1])
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    value_level = self.value_levels[row][0]
                    for entity in current_layout.children:
                        entity.size.set_value_level(value_level)
                    value_level = self.value_levels[row][1]
                    for entity in second_layout.children:
                        entity.size.set_value_level(value_level)
                else:
                    value_level = self.value_levels[row][2]
                    for entity in second_layout.children:
                        entity.size.set_value_level(value_level)
            self.count = (self.count + 1) % 6
        elif self.attr == "Color":
            if self.count == 0:
                all_value_levels = list(range(current_layout.entity_constraint["Color"][0],
                                              current_layout.entity_constraint["Color"][1] + 1))
                three_value_levels = np.random.choice(all_value_levels, 3, False)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                for entity in current_layout.children:
                    entity.color.set_value_level(self.value_levels[0][0])
                for entity in second_layout.children:
                    entity.color.set_value_level(self.value_levels[0][1])
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    value_level = self.value_levels[row][0]
                    for entity in current_layout.children:
                        entity.color.set_value_level(value_level)
                    value_level = self.value_levels[row][1]
                    for entity in second_layout.children:
                        entity.color.set_value_level(value_level)
                else:
                    value_level = self.value_levels[row][2]
                    for entity in second_layout.children:
                        entity.color.set_value_level(value_level)
            self.count = (self.count + 1) % 6
        else:
            raise ValueError("Unsupported attriubute")
        return second_aot