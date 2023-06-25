import re
import json
import asyncio
from .task import Task
from .common import get_first_letter
from datasets import load_dataset
from ..eval_llm import EvalLLM, LLMConfig

def try_parse(completion):
    letter = get_first_letter(completion)
    if letter:
        return letter.lower()
    else:
        return ""

class MPRETask(Task):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        dataset = load_dataset("andersonbcdefg/mpre", split="train")
        return dataset

    def prompt_template(self):
        return """Identify the correct answer to the following multiple-choice question about legal professional ethics. Your final answer should be a single letter (a, b, c, or d). Any other answer will be penalized. If unsure, provide your best guess.

QUESTION: {{ document.problem_statement }}

OPTIONS:
a. {{ document.candidate_answers[0] }}
b. {{ document.candidate_answers[1] }}
c. {{ document.candidate_answers[2] }}
d. {{ document.candidate_answers[3] }}

FINAL ANSWER (a, b, c, or d):""" 

    def score(self, docs, completions):
        correct_answers = ["abcd"[doc['correct_idx']] for doc in docs]
        preds = [try_parse(completion['text'].strip()) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)
        return {"accuracy": accuracy}

if __name__ == "__main__":
    task = MPRETask()
    asyncio.run(task.test())


