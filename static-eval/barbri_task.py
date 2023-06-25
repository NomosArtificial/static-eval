import asyncio
from .task import Task
from .common import get_first_letter
from datasets import load_dataset

def try_parse(completion):
    letter = get_first_letter(completion)
    if letter:
        return letter.lower()
    else:
        return ""
  
MAX_QUESTION_LENGTH_IN_CHARS = 12000

class BarbriTask(Task):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        dataset = load_dataset("LawInformedAI/duckai-barbri", split="train")
        dataset = dataset.map(
            lambda example: {k.replace(" ", "_").lower():v for k, v in example.items()},
            remove_columns=dataset.column_names
        ).map(
            lambda example: {"length": len(example['problem_statement']) + len(str(example['answer_candidates']))}
        )
        len_before_filter = len(dataset)
        dataset = dataset.filter(
           lambda example: example['length'] < MAX_QUESTION_LENGTH_IN_CHARS
        )
        print(f"After filtering out {len_before_filter - len(dataset)} questions greater than {MAX_QUESTION_LENGTH_IN_CHARS} characters, there are {len(dataset)} examples.")
        return dataset

    def prompt_template(self):
        return """Identify the correct answer for the following multiple-choice bar exam question. Your final answer should be a single letter (a, b, c, or d), with no explanation required. If unsure, provide your best guess, as any other answer will be considered incorrect.

QUESTION: {{ document.problem_statement }}

OPTIONS:
a. {{ document.answer_candidates[0] }}
b. {{ document.answer_candidates[1] }}
c. {{ document.answer_candidates[2] }}
d. {{ document.answer_candidates[3] }}

FINAL ANSWER (a, b, c, or d):""" 

    def score(self, docs, completions):
        correct_answers = [doc['final_answer'].lower() for doc in docs]
        preds = [try_parse(completion['text'].strip()) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)
        return {"accuracy": accuracy}

if __name__ == "__main__":
    asyncio.run(BarbriTask().test())


