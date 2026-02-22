"""Utils to work with open ai"""

from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


CLIENT = OpenAI()

def generate_baseline_prompt(question):
    prompt = f"""
        Generate an answer to this question:
        {question}

    """
    return prompt


def generate_policy_prompt(question, memory_entries):
    prompt = f"""
        You will get a question and you have to generate an answer this.
        You will receive questions and previous answers. The examples yield several
        answers to one questions, each with an advantage score. A high score means that
        an answer was successful. A low score means that the answer was less succesfull.
        Based on these examples derive how to answer the incoming question.

        _____________
        EXAMPLES:
        {memory_entries}


       Now generate an answer to this question with the derived answering policy.
        {question}
    """
    return prompt


def get_gpt_response(input_text: str, model: str = "gpt-5-mini-2025-08-07", temperature: float = 1.0) -> str:
    response = CLIENT.responses.create(
        model=model,
        input=input_text,
        temperature=temperature
    )
    return response.output_text


def get_gpt_rollouts(input_text: str, num_rollouts: int, model: str = "gpt-5-mini-2025-08-07", temperature: float = 1.0) -> List[str]:
    rollout_answers = []

    for _ in range(num_rollouts):
        answer = get_gpt_response(
            input_text=input_text,
            model=model,
            temperature=temperature,
        )
        rollout_answers.append(answer)
    
    return rollout_answers