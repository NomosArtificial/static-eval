"""
Dataset from: https://arxiv.org/pdf/1805.01217v2.pdf
(ZIP here: http://claudette.eui.eu/ToS.zip)
Available on HuggingFace at: https://huggingface.co/datasets/LawInformedAI/claudette_tos
"""
import asyncio
from .task import Task
from .common import get_first_letter
import math
import numpy as np
from datasets import load_dataset, concatenate_datasets

def try_parse(completion):
    letter = get_first_letter(completion)
    if letter:
        if letter.lower() == "y":
            return 1
        elif letter.lower() == "n":
            return 0
    return -1

class ToSTask(Task):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        dataset = load_dataset("LawInformedAI/claudette_tos", split="train")
        pos_examples = dataset.filter(lambda x: x['label'] == 1)
        neg_examples = dataset.filter(lambda x: x['label'] == 0)
        neg_subset = neg_examples.select(np.arange(len(pos_examples)) * int(math.floor(len(neg_examples) / len(pos_examples))))
        print("Created balanced dataset with {} positive and {} negative examples.".format(len(pos_examples), len(neg_subset)))
        return concatenate_datasets([pos_examples, neg_subset])
    
    def prompt_template(self):
        return """Article 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts defines an unfair contractual term as follows. A contractual term is unfair if: (1) it has not been individually negotiated; and (2) contrary to the requirement of good faith, it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer. The types of unfairness considered include: arbitration, unilateral change, content removal, jurisdiction, choice of law, limitation of liability, unilateral termination, and contract by using. Is the following sentence a potentially-unfair contractual term? Answer with a single word, 'Yes' or 'No'. If not sure, provide your best guess, as any other answer will be considered incorrect.
        
        SENTENCE: {{document.text}}

        ANSWER (yes/no):"""

    def score(self, docs, completions):
        correct_answers = [doc['label'] for doc in docs]
        preds = [try_parse(completion['text']) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)
        return {"accuracy": accuracy}

if __name__ == "__main__":
    asyncio.run(ToSTask().test())


