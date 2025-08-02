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


def parse_belief_state(raw_pred):
    """
    Convert raw model output string into normalized slot-value dictionary.
    Expected input: "hotel-price=cheap, hotel-area=centre"
    """
    belief = {}
    if not raw_pred or not isinstance(raw_pred, str):
        return belief

    items = [x.strip() for x in raw_pred.strip().split(",") if "=" in x]
    for item in items:
        try:
            slot, value = item.split("=", 1)
            slot = slot.strip()
            value = normalize_value(value)
            belief[slot] = value
        except ValueError:
            continue
    return belief


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
