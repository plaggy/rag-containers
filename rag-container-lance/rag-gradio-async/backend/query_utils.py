import openai
import gradio as gr
import os
import logging
import tiktoken

from enum import Enum
from typing import AsyncGenerator
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from jinja2 import Template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = open("/run/secrets/OPENAI_API_KEY").read().strip("\n")
HF_TOKEN = open("/run/secrets/HF_TOKEN").read().strip("\n")
PROMPT_TOKEN_LIMIT = int(os.getenv("PROMPT_TOKEN_LIMIT", 32768))

HF_TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL"))
# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
OAI_TOKENIZER = tiktoken.get_encoding("cl100k_base")

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


def truncate_context(
        tokenizer: AutoTokenizer | tiktoken.Encoding, 
        template: Template, 
        query: str, 
        docs: list[str]
    ) -> str:
    """
    This is an attempt to truncate the context to the preset limit in order to keep the postfix
    """
    # +2 for newlines
    preprompt_tokens = len(tokenizer.encode(template.render(query=query))) + 2
    context = "\n".join(docs)
    context = tokenizer.decode(
        # -6 to account for special tokens, just an approximation
        tokenizer.encode(context)[:PROMPT_TOKEN_LIMIT - preprompt_tokens - 6]
    )
    message = template.render(context=context, query=query)

    return message


def format_prompt_openai(template: Template, query: str, docs: list[str]) -> list[dict[str, str]]:
    """
    Formats the given message in a gpt-friendly format
    """
    message = truncate_context(OAI_TOKENIZER, template, query, docs)

    return [{"role": "user", "content": message}]


def format_prompt_hf(template: Template, query: str, docs: list[str]) -> str:
    """
    Formats the given message using HF chat template
    """
    message = truncate_context(HF_TOKENIZER, template, query, docs)
    message = HF_TOKENIZER.bos_token + "[INST] " + message + " [/INST]"

    return message


async def generate_hf(
        template: Template, query: str, docs: list[str], history: list[list]
    ) -> AsyncGenerator[str, None]:
    """
    Generate a sequence of tokens based on a given prompt and history using HF API.
    """
    formatted_prompt = format_prompt_hf(template, query, docs)
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


async def generate_openai(
        template: Template, query: str, docs: list[str], history: list[list]
    ) -> AsyncGenerator[str, None]:
    """
    Generate a sequence of tokens based on a given prompt and history using OpenAI API.
    """
    formatted_prompt = format_prompt_openai(template, query, docs)

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
    huggingface = generate_hf
    openai = generate_openai
