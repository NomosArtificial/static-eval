# Policy Change

**Contributor**: Sunny Gandhi

**Source**: [OPP-115](https://usableprivacy.org/data)

**License**: Creative Commons Attribution-NonCommercial License

**Task summary**: Does the clause describe if and how users will be informed about changes to the privacy policy?.

**Size (samples)**: 439

## Task Description

This is a binary classification task in which the LLM must answer the following annotation intent for clauses in privacy policies.

```text
Does the clause describe if and how users will be informed about changes to the privacy policy?
```

## Task Construction

This task was constructed from the [OPP-115 dataset](https://usableprivacy.org/data). Please see the [original paper](https://usableprivacy.org/static/files/swilson_acl_2016.pdf) for more details on construction. This dataset is class balanced.

# Column names
- `text`: clause from privacy policy
- `answer`: answer the annotation intent above as applied to the clause (Yes/No)