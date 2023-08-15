#!/usr/bin/env python
# coding: utf-8



import os
# get current working directory
os.getcwd()
import sys
sys.path.append("..")


from tasks.legalbench import LBTask
from datasets import load_dataset
from eval_llm import EvalLLM, LLMConfig
from datetime import datetime
import pandas as pd
import asyncio 


task_dir = "legalbench"
# gets all legalbench tasks
tasks = [
name
for name in os.listdir(task_dir)
if os.path.isdir(os.path.join(task_dir, name))
]
#tasks = tasks[11:17]
tasks = tasks[1:2]
print(tasks)



# creates a file to store the results in
header = True
results_filename = str(datetime.now()) + "_RESULTS.csv"
open(results_filename, "w+").close()


model_name = "NousResearch/Llama-2-7b-hf"

config = LLMConfig(
        model_name= model_name,
        max_new_tokens=6,
        temperature=0.0,
        rate_limit=600,
)



async def main():
    for task_name in tasks:
        task = LBTask(task_name, train = False)
        
        data = task.prepare_data()
        
        llm = EvalLLM(config, task.prompt_template())
        completions = await llm.run(data)
        
        metrics = task.score(data, completions)
        print(metrics)
        
        # append task result to result_file
        results_df = pd.DataFrame({"test": [task_name], "metrics": [metrics]})
        results_df.to_csv(results_filename, mode="a", index=False, header=header)
        header = False

asyncio.run(main())
