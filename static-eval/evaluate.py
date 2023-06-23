import fire
import yaml
import asyncio
from .tasks.registry import get_tasks
from .eval_llm import EvalLLM, LLMConfig

def evaluate(tasks: list[str], models: list[str], output_file: str = None):
    results = {task_name: {} for task_name in tasks}
    task_dict = get_tasks()
    for task_name in tasks:
        task = task_dict[task_name]()
        data = task.prepare_data()
        template = task.prompt_template()

        for model_name in models:
            print(f"Evaluating {model_name} on {task_name}...")
            config = LLMConfig(model_name=model_name, max_new_tokens=task.get_max_tokens(), temperature=0.0)
            llm = EvalLLM(config, template)
            completions = asyncio.run(llm.run(data, chunk_size = llm.recommended_chunk_size))
            metrics = task.score(data, completions)
            results[task_name][model_name] = metrics
    results = yaml.dump(results)
    print(results)
    # save to file
    if output_file:
        with open(output_file, "w") as f:
            f.write(results)

if __name__ == "__main__":
    fire.Fire(evaluate)