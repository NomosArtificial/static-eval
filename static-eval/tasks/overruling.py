import asyncio
from .task import Task
from .common import get_first_letter
from datasets import load_dataset

def try_parse(completion):
    letter = get_first_letter(completion)
    if letter:
        if letter.lower() == "y":
            return 1
        elif letter.lower() == "n":
            return 0
    return -1

class OverrulingTask(Task):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        dataset = load_dataset("LawInformedAI/overruling", split="train")
        return dataset
    
    def prompt_template(self):
        return """In law, an overruling sentence is a statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by the same or higher ranking court which establishes a different rule on the point of law involved. Is the following an overruling sentence? Answer with a single word, 'Yes' or 'No'. If not sure, provide your best guess, as any other answer will be considered incorrect.
        
        SENTENCE: {{document.sentence1}}

        ANSWER (Yes/No):"""

    def score(self, docs, completions):
        correct_answers = [doc['label'] for doc in docs]
        preds = [try_parse(completion['text']) for completion in completions]
        accuracy = sum([1 if pred == correct else 0 for pred, correct in zip(preds, correct_answers)]) / len(preds)
        return {"accuracy": accuracy}

if __name__ == "__main__":
    asyncio.run(OverrulingTask().test())


