from abc import ABC, abstractmethod
from typing import List, Dict, Iterable, Union, Any
from eval_llm import EvalLLM, LLMConfig


class Task(ABC):
    def __init__(self):
        pass

    def get_max_tokens(self) -> int:
        """
        Returns the maximum number of tokens needed to answer the task.
        """
        return 20

    @abstractmethod
    def prepare_data(self) -> Iterable[Dict]:
        """
        Returns an iterable (list, Dataset, etc.) of dicts, each is one data instance.
        """
        raise NotImplementedError

    @abstractmethod
    def prompt_template(self) -> str:
        """
        Prompt template must accept only one input variable, 'document'. The "document" is the entire
        data object ("document" isn't a key in the data instance, it represents the entire instance.)
        Fields of the document can be accessed using dot notation, e.g. {{ document.question }},
        {{ document.answers[0] }}, etc. This method should return a string, because how it is turned
        into a PromptTemplate depends on if the model is a ChatModel or vanilla LLM.
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self, docs: Iterable[Dict], completions: Iterable[Dict]
    ) -> Dict[str, Any]:
        """
        Returns one or more metrics for the task, as a dictionary.
        :param docs: the data instances
        :param completions: the completions generated by the model, dicts with key "text"
        """
        raise NotImplementedError

    async def test(self):
        """
        This method is used to test the task with gpt-3.5-turbo. Makes it unnecessary to write a separate
        test method for each task.
        """
        data = self.prepare_data()
        config = LLMConfig(
            model_name="gpt-3.5-turbo",
            max_new_tokens=self.get_max_tokens(),
            temperature=0.0,
        )
        llm = EvalLLM(config, self.prompt_template())
        completions = await llm.run(data)
        metrics = self.score(data, completions)
        print(metrics)
        return metrics
