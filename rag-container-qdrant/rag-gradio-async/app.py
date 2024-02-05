import logging
import gradio as gr
import os

from pathlib import Path
from time import perf_counter
from jinja2 import Environment, FileSystemLoader

from backend.query_llm import generate_hf, generate_openai
from backend.semantic_search import rerank, retrieve


proj_dir = Path(__file__).parent
# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader(proj_dir / 'templates'))

# Load the templates directly from the environment
template = env.get_template('template.j2')
template_html = env.get_template('template_html.j2')


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


async def bot(history, use_ranker, api_kind):
    top_k_rank = int(os.getenv("TOP_K_RANK"))
    top_k_retrieve = int(os.getenv("TOP_K_RETRIEVE"))
    query = history[-1][0] or ''

    if not query:
        raise gr.Error("The request is empty, please type something in")

    logger.info('Retrieving documents...')
    tic = perf_counter()
    if use_ranker:
        retrieved_docs = await retrieve(query, top_k_retrieve)
        documents = await rerank(query, retrieved_docs, top_k_rank)
    else:
        documents = await retrieve(query, top_k_rank)

    document_time = perf_counter() - tic
    logger.info(f'Finished Retrieving documents in {round(document_time, 2)} seconds...')

    # Create Prompt
    prompt = template.render(documents=documents, query=query)
    prompt_html = template_html.render(documents=documents, query=query)

    if api_kind == "HuggingFace":
         generate_fn = generate_hf
    elif api_kind == "OpenAI":
         generate_fn = generate_openai
    elif api_kind is None:
         raise ValueError("API name was not provided")
    else:
         raise ValueError(f"API {api_kind} is not supported")

    history[-1][1] = ""
    async for character in generate_fn(prompt, history[:-1]):
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

    cb = gr.Checkbox(label="Use cross-encoder", info="Rerank after retrieval?")
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
