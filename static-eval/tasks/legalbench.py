import fire
import requests

import pandas as pd

from .task import Task
from .common import get_first_letter
from ..eval_llm import EvalLLM, LLMConfig


def try_parse(completion):
    letter = get_first_letter(completion)

    if letter:
        if letter.lower() == 'y':
            return 'Yes'
        elif letter.lower() == 'n':
            return 'No'

    return completion


class LBTask(Task):
    def __init__(
        self,
        data_loc: str = './data/test.tsv',
        label_col: str = 'label',
        local: bool = True,
        prompt_loc: str = './data/base_prompt.txt'
    ):
        super().__init__()
        self.data_loc = data_loc
        self.prompt_loc = prompt_loc
        self.label_col = label_col
        self.local = local

    def prepare_data(self):
        raw_data = pd.read_csv(self.data_loc, sep='\t')
        data = [{col: raw_data[col][i] for col in list(raw_data.columns)} for i in range(len(raw_data))]

        return data

    def prompt_template(self):
        if self.local:
            with open(self.prompt_loc) as in_file: prompt_template = in_file.read()
        else:
            prompt_template = requests.get(self.prompt_loc).text

        return prompt_template.replace('{{', '{{document.')

    def score(self, docs, completions):
        correct_answers = [doc[self.label_col] for doc in docs]
        preds = [try_parse(completion['text']) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)

        return {'accuracy': accuracy}


async def test():
    task = LBTask()
    data = task.prepare_data()
    template = task.prompt_template()
    print(template)
    config = LLMConfig(model_name='gpt-3.5-turbo', max_new_tokens=20, temperature=0.0, rate_limit=2000)
    llm = EvalLLM(config, task.prompt_template())
    completions = await llm.run(data)
    metrics = task.score(data, completions)
    print(metrics)


if __name__ == '__main__':
    fire.Fire(test)
