import argparse
import os
import json
from tqdm import tqdm

from models.baseline_dst import T5DSTBaseline, LLM_DST_Baseline, HF_DST_GenericBaseline
from utils.evaluation import parse_belief_state, compute_joint_goal_accuracy, load_json


def load_dialogues(data_path):
    data = load_json(data_path)
    dialogues = []
    gold_states = []
    for dial in data:
        turns = dial["dialogue"]
        for turn_idx in range(len(turns)):
            history = turns[:turn_idx + 1]
            gold = {}
            for slot_info in turns[turn_idx].get("belief_state", []):
                for slot, value in slot_info["slots"]:
                    gold[slot] = value

            dialogues.append(history)
            gold_states.append(gold)
    return dialogues, gold_states


def get_model(model_type, model_id="google/flan-t5-base", fewshot_path=None):
    if model_type == "t5":
        return T5DSTBaseline()
    elif model_type == "gpt":
        return LLM_DST_Baseline()
    elif model_type == "hf":
        return HF_DST_GenericBaseline(model_id=model_id, fewshot_path=fewshot_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to dev/test JSON file", default="data/mw21/dev_dials.json")
    parser.add_argument("--model", type=str, choices=["t5", "gpt", "hf"], default="hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hf_model_id", type=str, help="Model ID from HuggingFace for HF mode", default="google/flan-t5-base")
    parser.add_argument("--fewshot_data", type=str, help="Path to training data JSON file for few-shot examples", default="data/mw21/train_dials.json")

    args = parser.parse_args()
    args.backup_path = f"results/predictions/{args.model}_{args.hf_model_id}.json"
    
    model = get_model(args.model, model_id=args.hf_model_id)
    dialogues, gold_states = load_dialogues(args.data_path)
    formatted_dialogues = [model.format_dialogue_history(turns) for turns in dialogues]

    predictions = []
    if os.path.exists(args.backup_path):
        with open(args.backup_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"Resuming from {len(predictions)} predictions in {args.backup_path}")

    start = len(predictions)
    pbar = tqdm(total=len(formatted_dialogues), initial=start)
    for i in range(start, len(formatted_dialogues), args.batch_size):
        batch = formatted_dialogues[i:i + args.batch_size]
        batch_preds = model.predict_belief_state_batch(batch)
        for j, pred in enumerate(batch_preds):
            dialogue_index = i + j
            entry = {
                "dialogue_index": dialogue_index,
                "dialogue_history": formatted_dialogues[dialogue_index],
                "raw_prediction": pred,
                "parsed_prediction": parse_belief_state(pred),
                "gold_state": gold_states[dialogue_index],
            }
            entry["match"] = entry["parsed_prediction"] == entry["gold_state"]
            predictions.append(entry)

        with open(args.backup_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

        pbar.update(len(batch_preds))
    pbar.close()

    parsed_preds = [parse_belief_state(p) for p in predictions]
    jga = compute_joint_goal_accuracy(parsed_preds, gold_states)
    print(f"Joint Goal Accuracy (JGA): {jga:.2%}")


if __name__ == "__main__":
    main()
