"""Module to assign rewards to rollout answers an calculate the advantage score.""" # This is an abstraction of GRPO - some elements are missing
from typing import List
import numpy as np

    
# inital policy is only based on answer length
def get_answer_length(answer_rollouts: List[str]) -> List[int]:
    return [len(answer.split())*-1 for answer in answer_rollouts]


def calculate_advantage_scores(rewards: List[int]) -> List[float]:
    reward_array = np.array(rewards)
    reward_mean = np.mean(reward_array)
    reward_std = np.std(reward_array)
    advantage_scores = (reward_array - reward_mean) / reward_std
    return list(advantage_scores)

def get_grpo_advantage_score(answer_rollouts: List[str]) -> List[float]:
    rewards = get_answer_length(answer_rollouts)
    advantage_scores = calculate_advantage_scores(rewards)
    return advantage_scores