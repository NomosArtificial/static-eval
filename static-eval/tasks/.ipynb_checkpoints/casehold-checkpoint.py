"""
CaseHOLD dataset from https://github.com/reglab/casehold.
Licensed under the Apache license.
"""
import math
import asyncio
from .task import Task
from .common import get_first_letter
from datasets import load_dataset

MAX_QUESTION_LENGTH_IN_CHARS = 12000
EXAMPLES_TO_KEEP = 1000

def try_parse(completion):
    letter = get_first_letter(completion)
    if letter:
        return letter.lower()
    else:
        return ""

class CaseHOLDTask(Task):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        dataset = load_dataset("casehold/casehold", split="train").map(
           lambda x: {"length": len(x["citing_prompt"]) + len(x['holding_0']) + len(x['holding_1']) + len(x['holding_2']) + len(x['holding_3']) + len(x['holding_4'])}
        )
        len_before_filter = len(dataset)
        dataset = dataset.filter(
           lambda x: x["length"] < MAX_QUESTION_LENGTH_IN_CHARS
        )
        print(f"After filtering out {len_before_filter - len(dataset)} questions greater than {MAX_QUESTION_LENGTH_IN_CHARS} characters, there are {len(dataset)} examples")
        subset = dataset.select(range(0, len(dataset), math.floor(len(dataset) / EXAMPLES_TO_KEEP))).select(range(EXAMPLES_TO_KEEP))
        print(f"Kept {len(subset)} examples")
        return subset

    def prompt_template(self):
        return """For the following excerpt of a judicial ruling, identify which of the holding statements makes sense to cite in support of the ruling. Your final answer should be a single letter (a, b, c, d, or e). Any other answer will be penalized. No explanation is required. If unsure, provide your best guess.

EXCERPT: {{ document.citing_prompt }}

HOLDING STATEMENTS:
a. {{ document.holding_0 }}
b. {{ document.holding_1 }}
c. {{ document.holding_2 }}
d. {{ document.holding_3 }}
e. {{ document.holding_4 }}

FINAL ANSWER (a, b, c, d, or e):""" 

    def score(self, docs, completions):
        correct_answers = ["abcde"[int(doc['label'])] for doc in docs]
        preds = [try_parse(completion['text']) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)
        return {"accuracy": accuracy}

if __name__ == "__main__":
    asyncio.run(CaseHOLDTask().test())


