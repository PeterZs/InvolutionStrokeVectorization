
import json
from typing import Dict, Any, List
import re
import os
import random
import string

def append_dict(filename: str, data: Dict[str, Any]) -> None:
    """
    Appends a dict to a JSONL file.
    
    :param filename: Path to the .jsonl file.
    :param data: Dictionary to append as a JSON line.
    """
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")


def read_dicts(filename: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file and returns a list of dicts.
    
    :param filename: Path to the .jsonl file.
    :return: A list of dictionaries, one for each line.
    """
    result = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue  # Skip blank lines
            result.append(json.loads(line))
    return result

def get_generated_count_for_class(out_dir: str, classname: str):
    """
    Scans the output directory for files matching the pattern:
    uuidv4_classname_count1_count2
    and returns the highest count1 for the given classname.
    """
    pattern = re.compile(rf"^{re.escape(classname)}_(\d+)_\d+_.+\..+")
    max_count1 = -1
    with os.scandir(out_dir) as entries:
        for entry in entries:
            if entry.is_file():
                match = pattern.match(entry.name)
                if match:
                    count1 = int(match.group(1))
                    if count1 > max_count1:
                        max_count1 = count1

    return max_count1

def delete_max(out_dir: str, classname: str, num: int):
    pattern = re.compile(rf"^{re.escape(classname)}_{num}_\d+_.+\..+")
    with os.scandir(out_dir) as it:
        for entry in it:
            if entry.is_file() and pattern.match(entry.name):
                try:
                    os.remove(entry.path)
                    print(f"Deleted: {entry.name}")
                except Exception as e:
                    print(f"Failed to delete {entry.name}: {e}")


def rand_chars(k=2):
    chars = random.choices(string.ascii_lowercase, k=k)
    return ''.join(chars)