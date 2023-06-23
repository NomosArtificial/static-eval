import itertools
from typing import List

def chunked_iterable(iterable, chunk_size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def prep_perf_data(d: List, p: List) -> List:
    for i, eg in enumerate(d):
        eg['id'] = str(i)
        eg['answers'] = {"text": [eg['answer']], "answer_start": [0]}
        p[i]['id'] = str(i)
        p[i]['prediction_text'] = p[i]['text']

    for _p in p:
        del _p['text']

    for eg in d:
        del eg['question']
        del eg['answer']

    return d, p