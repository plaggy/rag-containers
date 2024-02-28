import os
import logging
import gradio as gr

from pathlib import Path
from time import perf_counter
from jinja2 import Environment, FileSystemLoader
from typing import AsyncGenerator

from backend.query_utils import GenFunc
from backend.semantic_search import rerank, retrieve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K_RANK = int(os.getenv("TOP_K_RANK"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE"))

proj_dir = Path(__file__).parent
env = Environment(loader=FileSystemLoader(proj_dir / 'templates'))

template = env.get_template('template.j2')
template_html = env.get_template('template_html.j2')


def add_text(history: list[list], text: str) -> tuple[list[list], gr.Textbox]:
    history = [] if history is None else history
    history = history + [[text, None]]
    return history, gr.Textbox(value="", interactive=False)


async def bot(history: list[list], use_ranker: bool, api_kind: str) -> AsyncGenerator[tuple[list[list], str], None]:
    if not history or not history[-1][0]:
        raise gr.Warning("The request is empty, please type something in")
    
    query = history[-1][0]
    generate_fn = getattr(GenFunc, api_kind.lower())

    logger.info('Retrieving documents...')
    tic = perf_counter()
    if use_ranker:
        retrieved_docs = await retrieve(query, TOP_K_RETRIEVE)
        documents = await rerank(query, retrieved_docs, TOP_K_RANK)
    else:
        documents = await retrieve(query, TOP_K_RANK)
    document_time = perf_counter() - tic
    logger.info(f'Finished Retrieving documents in {round(document_time, 2)} seconds...')

    prompt_html = template_html.render(documents=documents, query=query)

    history[-1][1] = ""
    async for character in generate_fn(template, query, documents, history[:-1]):
        history[-1][1] = character
        yield history, prompt_html


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        avatar_images=('https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg',
                        'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'),
        bubble_full_width=False,
        show_copy_button=True,
        show_share_button=True,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        txt_btn = gr.Button(value="Submit text", scale=1)

    cb = gr.Checkbox(label="Use reranker", info="Rerank after retrieval?")
    api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace")

    prompt_html = gr.HTML()
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, cb, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, cb, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)


demo.queue()
demo.launch(debug=True, share=True)
