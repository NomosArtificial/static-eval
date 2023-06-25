import os
from datetime import datetime

import pandas as pd

# get current working directory
os.getcwd()
import sys
sys.path.append("..")
from legalbench import LBTask
from eval_llm import EvalLLM, LLMConfig
async def main():
    os.environ["OPENAI_API_KEY"] = "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"
    task_dir = "legalbench"
    # gets all legalbench tasks
    tasks = [
        name
        for name in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, name))
    ]
    # creates a file to store the results in
    header = True
    results_filename = str(datetime.now()) + "_RESULTS.csv"
    open(results_filename, "w+").close()

    for task_name in tasks:

        task = LBTask(task_name)
        data = task.prepare_data()
        config = LLMConfig(model_name="gpt-3.5-turbo", max_new_tokens=20, temperature=0.0, rate_limit=3500)
        llm = EvalLLM(config, task.prompt_template())
        completions = await llm.run(data)
        metrics = task.score(data, completions)
        print(metrics)
        # append task result to result_file
        results_df = pd.DataFrame(
            {"test": [task_name], "metrics": [metrics]}
        )

        results_df.to_csv(results_filename, mode="a", index=False, header=header)
        header = False

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
