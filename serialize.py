# -*- coding: utf-8 -*-


import json
import xml.etree.ElementTree as ET

import numpy as np

from const import META_STRUCTURE_FORMAT
from api import get_real_bbox, get_mask, rle_encode


def n_tree_serialize(aot):
    assert aot.is_pg
    ret = ""
    if aot.level == "Layout":
        return aot.name + "./"
    else:
        ret += aot.name + "."
        for child in aot.children:
            x = n_tree_serialize(child)
            ret += x
            ret += "."
        ret += "/"
    return ret


def serialize_aot(aot):
    """Meta Structure format
    META_STRUCTURE_FORMAT provided by const.py
    """
    n_tree = n_tree_serialize(aot)
    meta_structure = np.zeros(len(META_STRUCTURE_FORMAT), np.uint8)
    split = n_tree.split(".")
    for node in split:
        try:
            node_index = META_STRUCTURE_FORMAT.index(node)
            meta_structure[node_index] = 1
        except ValueError:
            continue
    return split, meta_structure


def serialize_rules(rule_groups):
    """Meta matrix format
    ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
    """
    meta_matrix = np.zeros((8, 9), np.uint8)
    counter = 0
    for rule_group in rule_groups:
        for rule in rule_group:
            if rule.name == "Constant":
                meta_matrix[counter, 0] = 1
            elif rule.name == "Progression":
                meta_matrix[counter, 1] = 1
            elif rule.name == "Arithmetic":
                meta_matrix[counter, 2] = 1
            else:
                meta_matrix[counter, 3] = 1
            if rule.attr == "Number/Position":
                meta_matrix[counter, 4] = 1
                meta_matrix[counter, 5] = 1
            elif rule.attr == "Number":
                meta_matrix[counter, 4] = 1
            elif rule.attr == "Position":
                meta_matrix[counter, 5] = 1
            elif rule.attr == "Type":
                meta_matrix[counter, 6] = 1
            elif rule.attr == "Size":
                meta_matrix[counter, 7] = 1
            else:
                meta_matrix[counter, 8] = 1
            counter += 1
    return meta_matrix, np.bitwise_or.reduce(meta_matrix)


def dom_problem(instances, all_column_rules):
    """
    instances: 14个上下文AOT + N个候选AOT (N >= 1)
    all_column_rules: 3个列规则组的列表 (用于 t=2, t=3, t=4)
    """
    data = ET.Element("Data")
    panels = ET.SubElement(data, "Panels")
    for i in range(len(instances)):
        panel = instances[i]
        if panel is None: continue

        panel_i = ET.SubElement(panels, "Panel")
        struct = panel.children[0]
        struct_i = ET.SubElement(panel_i, "Struct")
        struct_i.set("name", struct.name)
        for j in range(len(struct.children)):
            component = struct.children[j]
            component_j = ET.SubElement(struct_i, "Component")
            component_j.set("id", str(j))
            component_j.set("name", component.name)
            layout = component.children[0]
            layout_k = ET.SubElement(component_j, "Layout")
            layout_k.set("name", layout.name)
            layout_k.set("Number", str(layout.number.get_value_level()))
            layout_k.set("Position", json.dumps(layout.position.values))
            layout_k.set("Uniformity", str(layout.uniformity.get_value_level()))
            for l in range(len(layout.children)):
                entity = layout.children[l]
                entity_l = ET.SubElement(layout_k, "Entity")
                entity_bbox = entity.bbox
                entity_type = entity.type.get_value()
                entity_size = entity.size.get_value()
                entity_angle = entity.angle.get_value()
                entity_l.set("bbox", json.dumps(entity_bbox))
                entity_l.set("real_bbox",
                             json.dumps(get_real_bbox(entity_bbox, entity_type, entity_size, entity_angle)))
                entity_l.set("mask", rle_encode(get_mask(entity_bbox, entity_type, entity_size, entity_angle)))
                entity_l.set("Type", str(entity.type.get_value_level()))
                entity_l.set("Size", str(entity.size.get_value_level()))
                entity_l.set("Color", str(entity.color.get_value_level()))
                entity_l.set("Angle", str(entity.angle.get_value_level()))

    rules = ET.SubElement(data, "Rules")

    for i in range(len(all_column_rules)):
        rule_groups_for_col = all_column_rules[i]
        col_rules_i = ET.SubElement(rules, "Column_Rule_Set")
        col_rules_i.set("column_index", str(i + 2))

        for j in range(len(rule_groups_for_col)):
            rule_group_for_comp = rule_groups_for_col[j]
            comp_rule_j = ET.SubElement(col_rules_i, "Component_Rule_Group")
            comp_rule_j.set("component_id", str(j))

            for rule in rule_group_for_comp:
                rule_k = ET.SubElement(comp_rule_j, "Rule")
                rule_k.set("name", rule.name)
                rule_k.set("attr", rule.attr)
                # --- 新增：保存 rule.value ---
                rule_k.set("value", str(rule.value))
                # --- 新增结束 ---

    modified_attr = ET.SubElement(data, "Modified_attributes")

    num_context_panels = (3 * 5) - 1

    # --- 修复：使用动态循环 ---
    num_candidates = len(instances) - num_context_panels
    for i in range(num_candidates):
        # --- 修复结束 ---
        candidate = instances[num_context_panels + i]

        candidate_i = ET.SubElement(modified_attr, 'Candidate')
        candidate_i.set("id", str(i))

        for attr in candidate.modified_attr:
            attr_j = ET.SubElement(candidate_i, "Attribute")
            attr_j.set('component_id', str(attr[0]))
            attr_j.set('name', attr[1])

    return ET.tostring(data)