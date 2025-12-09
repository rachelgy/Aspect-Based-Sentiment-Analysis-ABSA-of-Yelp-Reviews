import ujson as json
from typing import Iterator, Dict

def stream_json(path: str) -> Iterator[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
