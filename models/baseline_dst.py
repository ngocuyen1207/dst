# models/baseline_dst.py

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import sys
from typing import List
from torch.utils.data import DataLoader
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

class Qwen3_DST_GenericBaseline:
    def __init__(self, model_id="Qwen/Qwen3-0.6B", temperature=0.0, max_tokens=512,fewshot_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    def predict_belief_state_batch(self, dialogue_histories: List[str]):
        # Create a DataLoader to process batches
        loader = DataLoader(dialogue_histories, batch_size=len(dialogue_histories), shuffle=False)

        contents, thinking_contents = [], []

        with torch.inference_mode():
            for batch_histories in loader:
                prompts = [self._format_prompt(history) for history in batch_histories]
                messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]

                # Apply chat template (if needed)
                texts = [
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False  # Disable unless using </think>
                    )
                    for messages in messages_batch
                ]

                model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                for input_ids, gen_ids in zip(model_inputs['input_ids'], generated_ids):
                    new_tokens = gen_ids[len(input_ids):].tolist()

                    # If you're using `</think>`, replace this section
                    thinking = ""
                    content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    thinking_contents.append(thinking)
                    contents.append(content)

        return contents, thinking_contents

    def _format_prompt(self, dialogue_history):
        return f"""You are an assistant that extracts belief states from user–system dialogues.

    Given a dialogue, return the belief state as a JSON dictionary with slot–value pairs that reflect the user’s current intent.

    Example 1:

    Dialogue:
    User: I want to find an expensive Indian restaurant in the south.
    System: What time would you like the reservation?

    Belief State:
    {{
        "restaurant-pricerange": "expensive",
        "restaurant-food": "Indian",
        "restaurant-area": "south"
    }}

    Example 2:

    Dialogue:
    User: I'm looking for a 4-star hotel in the north with free parking.
    System: Do you have a price range in mind?
    User: No, price doesn't matter.

    Belief State:
    {{
        "hotel-stars": "4",
        "hotel-area": "north",
        "hotel-parking": "yes"
    }}

    Example 3:

    Dialogue:
    User: I’d like to book a hotel in the west with free Wi-Fi.
    System: Do you have a preferred hotel?
    User: Yes, the Hamilton Lodge. For 2 people, 3 nights starting Saturday.

    Belief State:
    {{
        "hotel-name": "Hamilton Lodge",
        "hotel-area": "west",
        "hotel-internet": "yes",
        "hotel-book people": "2",
        "hotel-book stay": "3",
        "hotel-book day": "Saturday"
    }}

    Now extract the belief state from this dialogue:

    Dialogue:
    {dialogue_history}

    Belief State:
    """.strip()
