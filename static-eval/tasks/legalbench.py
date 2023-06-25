import requests
import pandas as pd

from tasks.task import Task
from tasks.common import get_first_letter
from eval_llm import EvalLLM, LLMConfig


def try_parse(completion):
    letter = get_first_letter(completion)

    if letter:
        if letter.lower() == "y":
            return "Yes"
        elif letter.lower() == "n":
            return "No"

    return completion


class LBTask(Task):
    def __init__(
        self,
        task: str,
        train: bool = False,
        local: bool = True,
    ):
        super().__init__()
        self.label_col = "label"
        self.data_loc = f"legalbench/{task}/train.tsv"
        self.prompt_loc = f"legalbench/{task}/base_prompt.txt"
        self.local = local

        if train:
            self.label_col = "label"
            self.data_loc = f"legalbench/{task}/train.tsv"

    def prepare_data(self):
        raw_data = pd.read_csv(self.data_loc, sep="\t")
        data = [
            {col: raw_data[col][i] for col in list(raw_data.columns)}
            for i in range(len(raw_data))
        ]

        return data

    def prompt_template(self):
        if self.local:
            try:
                with open(self.prompt_loc) as in_file:
                    prompt_template = in_file.read()
            except:
                prompt_template = "{{document}}"
        else:
            prompt_template = requests.get(self.prompt_loc).text

        return prompt_template.replace("{{", "{{document.")

    def score(self, docs, completions):
        correct_answers = [doc[self.label_col] for doc in docs]
        preds = [try_parse(completion["text"]) for completion in completions]
        accuracy = sum(
            [
                1 if pred == correct else 0
                for pred, correct in zip(preds, correct_answers)
            ]
        ) / len(preds)

        return {"accuracy": accuracy}
