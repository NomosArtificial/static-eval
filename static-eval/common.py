## This file is for common functions shared across tasks, such as output parsing functions
import re
import json

def get_first_letter(string):
    pattern = r"[a-zA-Z]"
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None

def try_parse_json(string):
    try:
        return json.loads(string)
    except:
        return None
