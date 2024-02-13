import openai
import gradio as gr
import logging
import os

from enum import Enum
from typing import AsyncGenerator

from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = open("/run/secrets/OPENAI_API_KEY").read().strip("\n")
HF_TOKEN = open("/run/secrets/HF_TOKEN").read().strip("\n")
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL"))

HF_CLIENT = AsyncInferenceClient(
    os.getenv("HF_URL"),
    token=HF_TOKEN
)
OAI_CLIENT = openai.AsyncClient(api_key=OPENAI_KEY)

HF_GENERATE_KWARGS = {
    'temperature': max(float(os.getenv("TEMPERATURE")), 1e-2),
    'max_new_tokens': int(os.getenv("MAX_NEW_TOKENS")),
    'top_p': float(os.getenv("TOP_P")),
    'repetition_penalty': float(os.getenv("REP_PENALTY")),
    'do_sample': bool(os.getenv("DO_SAMPLE")),
    'seed': int(os.getenv("SEED")),
}
   
OAI_GENERATE_KWARGS = {
    'temperature': max(float(os.getenv("TEMPERATURE")), 1e-2),
    'max_tokens': int(os.getenv("MAX_NEW_TOKENS")),
    'top_p': float(os.getenv("TOP_P")),
    'frequency_penalty': max(-2, min(float(os.getenv("FREQ_PENALTY")), 2)),
    'seed': int(os.getenv("SEED"))
}


def format_prompt_openai(message: str) -> list[dict[str, str]]:
    """
    Formats the given message in a gpt-friendly format
    """
    messages = [{"role": "user", "content": message}]
    return messages


def format_prompt_hf(message: str) -> str:
    """
    Formats the given message using HF chat template
    """
    messages = [{"role": "user", "content": message}]
    return TOKENIZER.apply_chat_template(messages, tokenize=False)


async def generate_hf(prompt: str, history: str) -> AsyncGenerator[str]:
    """
    Generate a sequence of tokens based on a given prompt and history using HF API.
    """
    formatted_prompt = format_prompt_hf(prompt)
    formatted_prompt = formatted_prompt.encode("utf-8").decode("utf-8")

    # history is not used yet
    try:
        stream = await HF_CLIENT.text_generation(
            formatted_prompt,
            **HF_GENERATE_KWARGS,
            stream=True, 
            details=True, 
            return_full_text=False
        )
        output = ""
        async for response in stream:
            output += response.token.text
            yield output

    except Exception as e:
        raise gr.Error(str(e))


async def generate_openai(prompt: str, history: str) -> AsyncGenerator[str]:
    """
    Generate a sequence of tokens based on a given prompt and history using OpenAI API.
    """

    formatted_prompt = format_prompt_openai(prompt)

    # history is not used yet
    try:
        stream = await OAI_CLIENT.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=formatted_prompt,
            **OAI_GENERATE_KWARGS, 
            stream=True
        )
        output = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                yield output

    except Exception as e:
        raise gr.Error(str(e))


class GenFunc(Enum):
    HF = generate_hf
    OPENAI = generate_openai
