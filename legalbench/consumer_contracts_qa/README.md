# consumer_contracts_qa

**Contributor**: Noam Kolt

**Source**: Noam Kolt

**License**: CC BY-NC 4.0

**Task summary**: Answer yes/no questions pertaining to the rights and obligations created by clauses in terms of services agreements.

**Size (samples)**: 400

## Task Description

The task consists of 400 yes/no questions relating to consumer contracts (specifically, online terms of service) - and is relevant to the legal skill of contract interpretation.

## Task Construction

The benchmark was originally developed for a [law review article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3844988) published in the Berkeley Technology Law Journal. See pages 94-102 of the article for additional information regarding the construction of the dataset.


## Files

- `train.tsv`: contains samples to be used as in-context demonstrations
- `test.tsv`: contains the evaluation set
- `base_prompt.txt`: a few-shot prompt that can be used to perform this task

## Data column names

In `train.tsv` and `test.tsv`, column names correspond to the following:
- `index`: sample identifier
- `contract`: a contract excerpt
- `question`: question to be answered
- `answer`: the answer to the question. Answers will be either `Yes` or `No`
