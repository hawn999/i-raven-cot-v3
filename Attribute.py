# -*- coding: utf-8 -*-


import numpy as np

from const import (ANGLE_MAX, ANGLE_MIN, ANGLE_VALUES, COLOR_MAX, COLOR_MIN,
                   COLOR_VALUES, NUM_MAX, NUM_MIN, NUM_VALUES, SIZE_MAX,
                   SIZE_MIN, SIZE_VALUES, TYPE_MAX, TYPE_MIN, TYPE_VALUES,
                   UNI_MAX, UNI_MIN, UNI_VALUES)


class Attribute:
    """Super-class for all attributes. This should not be instantiated.
    """

    def __init__(self, name):
        self.name = name
        self.level = "Attribute"
        # memory to store previous values
        self.previous_values = []

    def sample(self):
        pass

    def get_value(self):
        pass

    def set_value(self):
        pass

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


class Number(Attribute):

    def __init__(self, min_level=NUM_MIN, max_level=NUM_MAX):
        super(Number, self).__init__("Number")
        self.value_level = 0
        self.values = NUM_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=NUM_MIN, max_level=NUM_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(list(range(min_level, max_level + 1)))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            values = list(range(self.min_level, self.max_level + 1))
        else:
            values = list(range(min_level, max_level + 1))
        if not previous_values:
            available = set(values) - set(self.previous_values) - {self.value_level}
        else:
            available = set(values) - set(previous_values) - {self.value_level}

        if not available:  # 如果没有可用选项，放宽约束
            available = set(values) - {self.value_level}
        if not available:  # 极不可能，但作为保险
            available = set(values)

        new_idx = np.random.choice(list(available))
        return new_idx

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Type(Attribute):

    def __init__(self, min_level=TYPE_MIN, max_level=TYPE_MAX):
        super(Type, self).__init__("Type")
        self.value_level = 0
        self.values = TYPE_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=TYPE_MIN, max_level=TYPE_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(list(range(min_level, max_level + 1)))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            values = list(range(self.min_level, self.max_level + 1))
        else:
            values = list(range(min_level, max_level + 1))
        if not previous_values:
            available = set(values) - set(self.previous_values) - {self.value_level}
        else:
            available = set(values) - set(previous_values) - {self.value_level}

        if not available: available = set(values) - {self.value_level}
        if not available: available = set(values)

        new_idx = np.random.choice(list(available))
        return new_idx

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Size(Attribute):

    def __init__(self, min_level=SIZE_MIN, max_level=SIZE_MAX):
        super(Size, self).__init__("Size")
        self.value_level = 3
        self.values = SIZE_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=SIZE_MIN, max_level=SIZE_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(list(range(min_level, max_level + 1)))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            values = list(range(self.min_level, self.max_level + 1))
        else:
            values = list(range(min_level, max_level + 1))
        if not previous_values:
            available = set(values) - set(self.previous_values) - {self.value_level}
        else:
            available = set(values) - set(previous_values) - {self.value_level}

        if not available: available = set(values) - {self.value_level}
        if not available: available = set(values)

        new_idx = np.random.choice(list(available))
        return new_idx

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Color(Attribute):

    def __init__(self, min_level=COLOR_MIN, max_level=COLOR_MAX):
        super(Color, self).__init__("Color")
        self.value_level = 0
        self.values = COLOR_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=COLOR_MIN, max_level=COLOR_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(list(range(min_level, max_level + 1)))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            values = list(range(self.min_level, self.max_level + 1))
        else:
            values = list(range(min_level, max_level + 1))
        if not previous_values:
            available = set(values) - set(self.previous_values) - {self.value_level}
        else:
            available = set(values) - set(previous_values) - {self.value_level}

        if not available: available = set(values) - {self.value_level}
        if not available: available = set(values)

        new_idx = np.random.choice(list(available))
        return new_idx

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        if isinstance(value_level, np.int64):
            value_level = int(value_level)
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Angle(Attribute):

    def __init__(self, min_level=ANGLE_MIN, max_level=ANGLE_MAX):
        super(Angle, self).__init__("Angle")
        self.value_level = 3
        self.values = ANGLE_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self, min_level=ANGLE_MIN, max_level=ANGLE_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(list(range(min_level, max_level + 1)))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            values = list(range(self.min_level, self.max_level + 1))
        else:
            values = list(range(min_level, max_level + 1))
        if not previous_values:
            available = set(values) - set(self.previous_values) - {self.value_level}
        else:
            available = set(values) - set(previous_values) - {self.value_level}

        if not available: available = set(values) - {self.value_level}
        if not available: available = set(values)

        new_idx = np.random.choice(list(available))
        return new_idx

    def get_value_level(self):
        return self.value_level

    def set_value_level(self, value_level):
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Uniformity(Attribute):

    def __init__(self, min_level=UNI_MIN, max_level=UNI_MAX):
        super(Uniformity, self).__init__("Uniformity")
        self.value_level = 0
        self.values = UNI_VALUES
        self.min_level = min_level
        self.max_level = max_level

    def sample(self):
        self.value_level = np.random.choice(list(range(self.min_level, self.max_level + 1)))

    def sample_new(self):
        # Should not resample uniformity
        pass

    def set_value_level(self, value_level):
        self.value_level = np.clip(value_level, self.min_level, self.max_level)

    def get_value_level(self):
        return self.value_level

    def get_value(self, value_level=None):
        if value_level is None:
            value_level = self.value_level
        return self.values[value_level]


class Position(Attribute):
    """Position is a special case.
    """

    def __init__(self, pos_type, pos_list):
        super(Position, self).__init__("Position")
        # planar: [x_c, y_c, max_w, max_h]
        # angular: [x_c, y_c, max_w, max_h, x_r, y_r, omega]
        assert pos_type in ("planar", "angular")
        self.pos_type = pos_type
        self.values = pos_list
        self.value_idx = None
        self.isChanged = False

    def sample(self, num):
        length = len(self.values)
        if num > length:
            # 如果请求的数量大于可用槽位 (例如 num=9, length=4), 则使用所有槽位
            num = length
        self.value_idx = np.random.choice(list(range(length)), num, False)

    # --- 修复：使用 cwhy/i-raven 的鲁棒循环来防止死锁 ---
    def sample_new(self, num, previous_values=None):
        # Here sample new relies on probability
        length = len(self.values)
        if num > length: num = length  # 确保 num 不大于 length

        if not previous_values:
            constraints = self.previous_values
        else:
            constraints = previous_values

        # 尝试50次，如果50次都找不到新组合（极不可能，除非 num=length）
        # 就跳出循环并返回最后一次尝试
        for _ in range(50):
            finished = True
            new_value_idx = np.random.choice(length, num, False)
            if set(new_value_idx) == set(self.value_idx):
                continue
            for previous_value in constraints:
                if set(new_value_idx) == set(previous_value):
                    finished = False
                    break
            if finished:
                break
        return new_value_idx

    # --- 修复结束 ---

    def sample_add(self, num):
        ret = []
        available = set(range(len(self.values))) - set(self.value_idx)
        num_to_sample = min(num, len(available))  # 确保采样数不超过可用数
        if num_to_sample == 0:
            return ret

        idxes_2_add = np.random.choice(list(available), num_to_sample, False)
        for index in idxes_2_add:
            self.value_idx = np.insert(self.value_idx, 0, index)
            ret.append(self.values[index])
        return ret

    def get_value_idx(self):
        return self.value_idx

    def set_value_idx(self, value_idx):
        self.value_idx = value_idx

    def get_value(self, value_idx=None):
        if value_idx is None:
            value_idx = self.value_idx
        ret = []
        for idx in value_idx:
            ret.append(self.values[idx])
        return ret

    def remove(self, bbox):
        idx = self.values.index(bbox)
        np_idx = np.where(self.value_idx == idx)[0][0]
        self.value_idx = np.delete(self.value_idx, np_idx)