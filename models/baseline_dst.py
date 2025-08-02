# models/baseline_dst.py

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import sys
sys.path.append('.')
from utils.evaluation import load_json

class LLM_DST_Baseline:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def predict_belief_state_batch(self, dialogue_histories):
        results = []
        for history in dialogue_histories:
            result = self.predict_single(history)
            results.append(result)
        return results

    def predict_single(self, formatted_history):
        prompt = f"""
Given the following dialogue history, extract the current belief state.
Return it as a list of 'domain-slot = value' pairs, separated by commas.
Only include slots that have been explicitly mentioned.

Dialogue:
{formatted_history}

Belief State:
"""
        message = HumanMessage(content=prompt.strip())
        response = self.llm([message])
        return response.content.strip()

    def format_dialogue_history(self, turns):
        history = ""
        for turn in turns:
            if turn.get("system_transcript"):
                history += f"System: {turn['system_transcript']}\n"
            if turn.get("transcript"):
                history += f"User: {turn['transcript']}\n"
        return history.strip()


class HF_DST_GenericBaseline:
    def __init__(self, model_id="google/flan-t5-base", temperature=0.0, max_tokens=512,fewshot_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_tokens = max_tokens
        
        self.fewshot_examples = []
        if fewshot_path:
            self.fewshot_examples = self.extract_fewshot_examples(fewshot_path)


    def format_dialogue_history(self, turns):
        history = ""
        for turn in turns:
            if turn.get("system_transcript"):
                history += f"System: {turn['system_transcript']}\n"
            if turn.get("transcript"):
                history += f"User: {turn['transcript']}\n"
        return history.strip()

    def predict_belief_state_batch(self, dialogue_histories):
        prompts = [
            self._format_prompt(history) for history in dialogue_histories
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=self.max_tokens)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs
    

    def extract_fewshot_examples(self, data_path, num_examples=3):
        from random import shuffle
        from utils.evaluation import flatten_belief_state

        data = load_json(data_path)
        shuffle(data)
        examples = []

        for dialogue in data:
            turns = dialogue["dialogue"]
            for i, turn in enumerate(turns):
                if "belief_state" not in turn or not turn["belief_state"]:
                    continue

                # Format the dialogue history up to this turn
                history = turns[:i+1]
                formatted_history = self.format_dialogue_history(history)

                # Flatten belief state into 'domain-slot = value' pairs
                flat_bs = flatten_belief_state(turn["belief_state"])
                if not flat_bs:
                    continue

                bs_str = ", ".join([f"{k} = {v}" for k, v in flat_bs.items()])
                examples.append((formatted_history, bs_str))

                if len(examples) >= num_examples:
                    return examples
        return examples


    def _format_prompt(self, dialogue_history):
        fewshot_text = ""
        for i, (history, belief_state) in enumerate(self.fewshot_examples):
            fewshot_text += f"\n### Example {i+1}\nDialogue:\n{history}\n\nBelief State:\n{belief_state}\n"

        return f"""Here is a dialogue between a user and a system:
{dialogue_history}

Extract the belief state from the dialogue.
Return it as a json, for example:

{{
    "hotel-area": "",
    "restaurant-food": "",
    "train-departure": "",
}}

""".strip()


class T5DSTBaseline:
    def __init__(self, model_name="t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def format_dialogue_history(self, turns):
        history = ""
        for turn in turns:
            if turn.get("system_transcript"):
                history += f"System: {turn['system_transcript']}\n"
            if turn.get("transcript"):
                history += f"User: {turn['transcript']}\n"
        return history.strip()

    def predict_belief_state_batch(self, dialogue_histories, max_length=128):
        inputs = self.tokenizer(dialogue_histories, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
