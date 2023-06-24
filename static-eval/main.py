import os
import sys
from datasets import load_dataset
from tasks.legalbench import LBTask
from eval_llm import EvalLLM, LLMConfig
import asyncio

# changed a lot of imports


async def main():
    os.getcwd()
    sys.path.append("..")

    os.environ["OPENAI_API_KEY"] = "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"

    config = LLMConfig(
        model_name="gpt-3.5-turbo", max_new_tokens=20, temperature=0.0, rate_limit=3500
    )

    task_dir = "legalbench"
    # gets all legalbench tasks
    tasks = [
        name
        for name in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, name))
    ]

    for task in tasks:
        task = LBTask(task)
        data = task.prepare_data()

        llm = EvalLLM(config, task.prompt_template())
        completions = await llm.run(data)
        metrics = task.score(data, completions)

        print(metrics)


if __name__ == "__main__":
    asyncio.run(yo())
