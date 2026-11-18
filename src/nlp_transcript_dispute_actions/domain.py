import json
from pathlib import Path

def load_facts(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def summarize(facts):
    return [reviewer['id'] for reviewer in facts['reviewers'] if reviewer['authorized']]
