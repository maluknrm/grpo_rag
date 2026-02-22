import os
import pandas as pd
from dotenv import load_dotenv
from grpo_rag.utils.openai_utils import (
    get_gpt_response, get_gpt_rollouts, generate_baseline_prompt, generate_policy_prompt
)
from grpo_rag.utils.constants import DATA_PATH, MEMORY_BUFFER_PATH
from grpo_rag.utils.grpo_utils import get_grpo_advantage_score
from grpo_rag.utils.memory_buffer_utils import (
    append_memory_entry, 
    sample_rollouts, 
    append_baseline_result_entry, 
    append_policy_result_entry
)
from grpo_rag.utils.data_utils import DataLoader

load_dotenv()

def main():
    print("Hello from grpo-rag!")
    num_samples = 3
    seed = 42
    num_rollout_answers = 2

    rollout_samples = 1

    print("set up test train")

    data_loader = DataLoader(sample=num_samples, random_state=seed)
    X_train, X_test, y_train, y_test = data_loader.get_train_test_set()

    print('Generate memory buffer')
    for i in range(len(X_train)):
        print(f'train iteration {i} out of {len(X_train)}')
        question = X_train[i]
        gold_answer = y_train[i]

        question_prompt = generate_baseline_prompt(question)
        rollout_answers = get_gpt_rollouts(question_prompt, num_rollout_answers)
        advantage_scores = get_grpo_advantage_score(rollout_answers)
        append_memory_entry(question, rollout_answers, advantage_scores, gold_answer)
    
    print('testing set up')
    for i in range(len(X_test)):
        print(f'test iteration {i} out of {len(X_test)}')
        question = X_test[i]
        gold_answer = y_test[i]

        baseline_question_prompt = generate_baseline_prompt(question)
        baseline_answer = get_gpt_response(baseline_question_prompt)
        append_baseline_result_entry(question, baseline_answer, gold_answer)


        policy_rollouts = sample_rollouts(MEMORY_BUFFER_PATH, rollout_samples, seed)
        policy_question_prompt = generate_policy_prompt(question, policy_rollouts)
        policy_answer = get_gpt_response(policy_question_prompt)
        append_policy_result_entry(question, policy_answer, gold_answer)




if __name__ == "__main__":
    main()
