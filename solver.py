# -*- coding: utf-8 -*-


import numpy as np
import copy


def solve(rule_groups, context, candidates, n_columns=5):
    """
    Search-based Heuristic Solver that finds the candidate that perfectly satisfies all rules.
    """
    # The context contains all panels except the last one.
    # The first (n_columns * 2) panels are the first two full rows.
    # The remaining panels are from the last row.
    last_row_context_panels = context[2 * n_columns:]

    for i, candidate in enumerate(candidates):
        is_perfect_match = True

        # Reconstruct the full last row with the current candidate
        reconstructed_last_row = last_row_context_panels + [candidate]

        for rule_group in rule_groups:
            # Check if this rule group is perfectly satisfied by the reconstructed row
            if not check_full_row_rules(rule_group, reconstructed_last_row, n_columns):
                is_perfect_match = False
                break  # This candidate is wrong, move to the next one

        if is_perfect_match:
            # Found the candidate that satisfies all rules. This must be the answer.
            return i

    # Fallback: This part should not be reached if the logic is perfect.
    return np.random.choice(len(candidates))


def check_full_row_rules(rule_group, row_panels, n_columns):
    """
    Checks if an entire row of panels strictly follows all rules in a rule_group.
    """
    for t in range(2, len(row_panels)):
        sub_context = row_panels[t - 2:t]
        sub_candidate = row_panels[t]

        # Check Number/Position rule
        if not check_num_pos(rule_group[0], sub_context, sub_candidate):
            return False

        # Check other entity attribute rules
        for rule in rule_group[1:]:
            regenerate = ("Number" in rule_group[0].attr or rule_group[0].name == "Arithmetic")
            if not check_entity(rule, sub_context, sub_candidate, rule.attr, regenerate):
                return False
    return True


def check_num_pos(rule_num_pos, context, candidate):
    """Checks the Number/Position rule for a single step: context -> candidate."""
    component_idx = rule_num_pos.component_idx
    context_layout_1 = context[0].children[0].children[component_idx].children[0]
    context_layout_2 = context[1].children[0].children[component_idx].children[0]
    candidate_layout = candidate.children[0].children[component_idx].children[0]

    if rule_num_pos.name == "Constant":
        num1 = context_layout_1.number.get_value()
        num2 = context_layout_2.number.get_value()
        num_cand = candidate_layout.number.get_value()
        return num1 == num2 and num2 == num_cand

    elif rule_num_pos.name == "Progression":
        if rule_num_pos.attr == "Number":
            num_1_lvl = context_layout_1.number.get_value_level()
            num_2_lvl = context_layout_2.number.get_value_level()
            candidate_num_lvl = candidate_layout.number.get_value_level()
            return (num_2_lvl - num_1_lvl) == (candidate_num_lvl - num_2_lvl) and (
                        num_2_lvl - num_1_lvl) == rule_num_pos.value
        else:  # Position
            if context_layout_1.number.get_value() != context_layout_2.number.get_value() or \
                    context_layout_2.number.get_value() != candidate_layout.number.get_value():
                return False  # Position progression requires same number of entities

            # If there are no entities, it's trivially true
            if context_layout_1.number.get_value() == 0:
                return True

            pos_1 = context_layout_1.position.get_value_idx()
            pos_2 = context_layout_2.position.get_value_idx()
            candidate_pos = candidate_layout.position.get_value_idx()
            most_num = len(candidate_layout.position.values)
            diff = rule_num_pos.value

            # This check must be exact. set() loses order and is not suitable for progression.
            # We sort to make the comparison canonical.
            expected_pos_2 = sorted([(p + diff) % most_num for p in pos_1])
            expected_pos_cand = sorted([(p + diff) % most_num for p in pos_2])

            return sorted(pos_2) == expected_pos_2 and sorted(candidate_pos) == expected_pos_cand


    elif rule_num_pos.name == "Arithmetic":
        mode = rule_num_pos.value
        if rule_num_pos.attr == "Number":
            num_1 = context_layout_1.number.get_value()
            num_2 = context_layout_2.number.get_value()
            candidate_num = candidate_layout.number.get_value()
            if mode > 0:
                return candidate_num == num_1 + num_2
            else:
                return candidate_num == abs(num_1 - num_2)  # Generation uses abs() for subtraction
        else:  # Position
            pos_1 = set(context_layout_1.position.get_value_idx())
            pos_2 = set(context_layout_2.position.get_value_idx())
            candidate_pos = set(candidate_layout.position.get_value_idx())
            if mode > 0:
                return candidate_pos == (pos_1 | pos_2)
            else:
                return candidate_pos == (pos_1 - pos_2)

    return True


def check_entity(rule, context, candidate, attr, regenerate):
    """Checks an entity-level rule for a single step: context -> candidate."""
    component_idx = rule.component_idx
    context_layout_1 = context[0].children[0].children[component_idx].children[0]
    context_layout_2 = context[1].children[0].children[component_idx].children[0]
    candidate_layout = candidate.children[0].children[component_idx].children[0]

    num1 = context_layout_1.number.get_value()
    num2 = context_layout_2.number.get_value()
    num_cand = candidate_layout.number.get_value()

    if regenerate:
        # If entities are supposed to be regenerated (due to Number rule),
        # we can't check entity-level rules. So we just assume it's correct.
        return True

    # If numbers are not constant, entity rules cannot be checked one-to-one
    if not (num1 == num2 and num2 == num_cand):
        return False

    # If panels are empty, rule holds trivially
    if num1 == 0:
        return True

    # All entities within a panel must have the same attribute value for these rules.
    if not check_consistency(context[0], attr, component_idx) or \
            not check_consistency(context[1], attr, component_idx) or \
            not check_consistency(candidate, attr, component_idx):
        return False

    val_1 = getattr(context_layout_1.children[0], attr.lower()).get_value_level()
    val_2 = getattr(context_layout_2.children[0], attr.lower()).get_value_level()
    candidate_val = getattr(candidate_layout.children[0], attr.lower()).get_value_level()

    if rule.name == "Constant":
        return val_1 == val_2 and val_2 == candidate_val
    elif rule.name == "Progression":
        return (val_2 - val_1) == (candidate_val - val_2) and (val_2 - val_1) == rule.value
    elif rule.name == "Arithmetic":
        # This logic MUST exactly match the generation logic in Rule.py
        if rule.value > 0:  # Addition
            expected_val = (val_1 + val_2) if attr == "Color" else (val_1 + val_2 + 1)
            return candidate_val == expected_val
        else:  # Subtraction
            expected_val = (val_1 - val_2) if attr == "Color" else (val_1 - val_2 - 1)
            return candidate_val == expected_val
    return True


def check_consistency(panel_aot, attr, component_idx):
    """Checks if all entities in a panel have the same value for a given attribute."""
    layout = panel_aot.children[0].children[component_idx].children[0]
    if not layout.children:
        return True  # Trivial consistency for empty panel
    attr_name = attr.lower()
    first_val = getattr(layout.children[0], attr_name).get_value_level()
    for entity in layout.children[1:]:
        if getattr(entity, attr_name).get_value_level() != first_val:
            return False
    return True