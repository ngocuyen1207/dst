# utils/evaluation.py

import json


def parse_belief_state(belief_text):
    """
    Parse belief state text to dict format. Expect format like:
    "domain-slot1 = value1, domain-slot2 = value2"
    """
    belief = {}
    if not belief_text.strip():
        return belief
    for pair in belief_text.split(","):
        if "=" in pair:
            slot, value = pair.split("=", 1)
            belief[slot.strip().lower()] = value.strip().lower()
    return belief

def compute_joint_goal_accuracy(predictions, references):
    """
    predictions: list of dicts (predicted belief states)
    references: list of dicts (ground truth belief states)
    """
    assert len(predictions) == len(references), "Mismatched prediction and reference count"
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
    return correct / total

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def flatten_belief_state(belief_state):
    flat = {}
    for entry in belief_state:
        for slot, value in entry.get("slots", []):
            flat[slot] = value
    return flat
