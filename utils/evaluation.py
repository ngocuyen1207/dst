import json
from sklearn.metrics import precision_recall_fscore_support


def normalize_value(value):
    value = value.strip().lower()
    mapping = {
        "centre": "center",
        "yes": "yes", "true": "yes",
        "no": "no", "false": "no",
        "none": "",
        "": "",
    }
    return mapping.get(value, value)


import json
import re

def parse_belief_state(pred: str) -> dict:
    """
    Extract and parse the belief state JSON from a prediction string.
    
    This handles:
    - <think>...</think> wrapping
    - extra text before or after the JSON
    - simple JSON decoding errors
    """
    # Step 1: Remove <think> tags if present
    pred = re.sub(r"</?think>", "", pred, flags=re.IGNORECASE).strip()

    # Step 2: Find the first JSON object using regex
    json_match = re.search(r'\{.*?\}', pred, re.DOTALL)
    if not json_match:
        return {}  # No JSON found

    json_str = json_match.group()

    # Step 3: Try to parse the JSON
    try:
        belief_state = json.loads(json_str)
    except json.JSONDecodeError:
        return {}  # Invalid JSON

    # Step 4: Normalize keys and values
    cleaned_belief_state = {}
    for k, v in belief_state.items():
        k = k.strip().lower()
        v = v.strip().lower() if isinstance(v, str) else v
        cleaned_belief_state[k] = v

    return cleaned_belief_state


def compute_joint_goal_accuracy(pred_beliefs, gold_beliefs):
    """
    Exact match per turn
    """
    correct = sum([p == g for p, g in zip(pred_beliefs, gold_beliefs)])
    total = len(gold_beliefs)
    return correct / total if total > 0 else 0.0


def compute_turn_accuracy(pred_beliefs, gold_beliefs):
    """
    Turn accuracy = (# correct slot predictions) / (# total slots)
    """
    total_slots = 0
    correct_slots = 0
    for pred, gold in zip(pred_beliefs, gold_beliefs):
        all_slots = set(pred.keys()).union(gold.keys())
        for slot in all_slots:
            pred_value = pred.get(slot, "")
            gold_value = gold.get(slot, "")
            if pred_value == gold_value:
                correct_slots += 1
            total_slots += 1
    return correct_slots / total_slots if total_slots > 0 else 0.0


def compute_micro_f1(pred_beliefs, gold_beliefs):
    """
    Compute micro-averaged precision/recall/F1 over all slot-value pairs.
    Treats each slot=value as a separate label.
    """
    gold_items = []
    pred_items = []

    for pred, gold in zip(pred_beliefs, gold_beliefs):
        gold_items += [f"{k}={v}" for k, v in gold.items()]
        pred_items += [f"{k}={v}" for k, v in pred.items()]

    labels = list(set(gold_items + pred_items))
    y_true = [label in gold_items for label in labels]
    y_pred = [label in pred_items for label in labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return precision, recall, f1


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
