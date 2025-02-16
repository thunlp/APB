
import json


def read_jsonl(filename, num_lines=-1):
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i == num_lines:
                break
    return lines


def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
