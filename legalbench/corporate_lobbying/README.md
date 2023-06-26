# corporate_lobbying 
 **Contributor**: John Nay
 
 **Source**: John Nay
 
 **License**: [CC By 4.0](https://creativecommons.org/licenses/by/4.0/)
 
 **Task summary**: Predict if a proposed bill is relevant to a company.
 
 **Size (samples)**: 500
 
 ## Task Description
 
 This task measures LLM ability to predict whether a proposed bill is relevant to a company.

 ## Files

- `train.tsv`: contains samples to be used as in-context demonstrations
- `test.tsv`: contains the evaluation set
- `base_prompt.txt`: a few-shot prompt that can be used to perform this task

## Data column names

In `train.tsv` and `test.tsv`, column names correspond to the following:
- `index`: sample identifier
 - `bill_title`: title of bill
 - `bill_summary`: summary of bill
 - `company_name`: name of company
 - `company_description`: description of company
 - `label`: whether the bill is relevant ("Yes") or not ("No")
