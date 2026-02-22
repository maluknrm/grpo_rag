"""Utils to manage memory buffer"""

import json
from pathlib import Path
from typing import TypedDict, List
import random
from grpo_rag.utils.constants import (
    MEMORY_BUFFER_PATH, RESULTS_PATH, RESULTS_BASELINE_FILE, RESULTS_POLICY_FILE
)

class MemoryEntry(TypedDict):
    query: str
    answers: List[str]
    advantage_scores: List[float]
    gold_answer: str | None = None

class ResultEntry(TypedDict):
    query: str
    answer: str
    gold_answer: str | None = None


def create_jsonl(filepath, records):
    """
    Create a JSON Lines file from a list of dictionaries.
    
    :param filepath: Path to the file
    :param records: List of dictionaries
    """
    filepath = Path(filepath)

    with filepath.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def append_memory_entry(query, answers, advantage_scores, gold_answer = None):
    memory_entry = MemoryEntry(
        query=query,
        answers=answers,
        advantage_scores=advantage_scores,
        gold_answer=gold_answer
    )
    append_jsonl(MEMORY_BUFFER_PATH, memory_entry)

def append_baseline_result_entry(query, answer, gold_answer = None):
    memory_entry = ResultEntry(
        query=query,
        answer=answer,
        gold_answer=gold_answer
    )
    append_jsonl(RESULTS_PATH + RESULTS_BASELINE_FILE, memory_entry)

def append_policy_result_entry(query, answer, gold_answer = None):
    memory_entry = ResultEntry(
        query=query,
        answer=answer,
        gold_answer=gold_answer
    )
    append_jsonl(RESULTS_PATH + RESULTS_POLICY_FILE, memory_entry)

def append_jsonl(filepath, record):
    """
    Append a single dictionary as a new JSON line.
    
    :param filepath: Path to the file
    :param record: Dictionary to append
    """
    filepath = Path(filepath)

    with filepath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(filepath):
    """
    Read a JSON Lines file into a list of dictionaries.
    
    :param filepath: Path to the file
    :return: List of dictionaries
    """
    filepath = Path(filepath)
    records = []

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records

def sample_rollouts(filepath, n, seed=None):
    """
    Sample n random records from a JSON Lines file.

    :param filepath: Path to file
    :param n: Number of samples
    :param seed: Optional random seed
    :return: List of sampled dictionaries
    """
    if seed is not None:
        random.seed(seed)

    records = read_jsonl(filepath)

    if n > len(records):
        raise ValueError(f"Requested {n} samples, but file only contains {len(records)} records.")

    return random.sample(records, n)