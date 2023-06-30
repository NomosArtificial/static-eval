# Static Evaluation of the Legal Reasoning and Legal Knowledge of Large Language Models

This repo contains LegalBench's training set along with scripts for evaluating LLMs on that data. 

The functions needed for eval can be found (here)[https://github.com/NomosArtificial/static-eval/blob/main/static-eval/task_utils.py].


## Notebooks and Scripts

- Example of running eval on the training set (including OpenAI APIs, and using Modal and Baseten for inference on open-source LLMs):
https://github.com/NomosArtificial/static-eval/blob/main/static-eval/eval_notebook.ipynb
- Example of script for hosting model on Modal.com for inference (follows https://modal.com/docs/guide/ex/falcon_gptq):
https://github.com/NomosArtificial/static-eval/blob/main/inference_scripts/modal_falcon7B.py
- Tutorial for baseten deployment of Falcon:
https://www.baseten.co/blog/deploy-falcon-40b-on-baseten

