# README

This is a access-by-url only DropBox folder which contains tasks and documentation for LegalBench. It has been made available in this format for the law x llm hackathon. We intend to release the full version of the benchmark in July 2023.

This folder contains data for current LegalBench tasks. Each subfolder corresponds to a different task. There are also two other scripts:

- `demo.ipynb`: a jupyter notebook with sample code for loading individual task data and generating prompts.
- `utils.py`: contains utility functions to help load data and prompts.


Each task subfolder contains the following files (at minimum):

- `README.md`: a ReadMe describing the task, data schema, and construction process. We note that this README contains less background regarding the task than the full task description provided in the paper. 
- `train.tsv`: a tab-separated file containing "train" samples for the task. We recommend using these samples as in-context demonstrations, for standardization purposes.
- `test.tsv`: a tab-separated file containing "test" samples for the task.
- `base_prompt.txt`: a prompt template to use for the task. 

Prompt templates contain column names in double brackets. When generating a prompt for a particular sample, the column names in brackets are replaced with sample's values for those columns. A prompt template may take the following form:

```text
Does the following clause mention the right for one party to audit another?

Clause: {{text}}
Answer:
```

For a sample represented by the dictionary: 

```text
{
    "text": "The clause permits one party to check and inspect the books of the other."
    "answer": "Yes"
}
```

We would generate the following prompt from this template:

```text
Does the following clause mention the right for one party to audit another?

Clause: The clause permits one party to check and inspect the books of the other.
Answer:
```

For several tasks, our paper describes generating additional prompts. These are also provided as `.txt` files.