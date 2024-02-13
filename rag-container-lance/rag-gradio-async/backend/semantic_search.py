import logging
import json
import gradio as gr
import numpy as np
import lancedb
import os
from huggingface_hub import AsyncInferenceClient


# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# db
TABLE_NAME = "docs"
TEXT_COLUMN = "text"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NPROBES = int(os.getenv("NPROBES"))
REFINE_FACTOR = int(os.getenv("REFINE_FACTOR"))

retriever = AsyncInferenceClient(model=os.getenv("EMBED_URL") + "/embed")
reranker = AsyncInferenceClient(model=os.getenv("RERANK_URL") + "/rerank")

db = lancedb.connect("/usr/src/.lancedb")
tbl = db.open_table(TABLE_NAME)


async def retrieve(query: str, k: int) -> list[str]:
    """
    Retrieve top k items with RETRIEVER
    """
    resp = await retriever.post(
        json={
            "inputs": query,
            "truncate": True
        }
    )
    try:
        query_vec = json.loads(resp)[0]
    except:
        raise gr.Error(resp.decode())
    
    documents = tbl.search(
        query=query_vec
    ).nprobes(NPROBES).refine_factor(REFINE_FACTOR).limit(k).to_list()
    documents = [doc[TEXT_COLUMN] for doc in documents]

    return documents


async def rerank(query: str, documents: list[str], k: int) -> list[str]:
    """
    Rerank items returned by RETRIEVER and return top k
    """
    scores = []
    for i in range(int(np.ceil(len(documents) / BATCH_SIZE))):
        resp = await reranker.post(
            json={
                "query": query,
                "texts": documents[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                "truncate": True
            }
        )
        try:
            batch_scores = json.loads(resp)
            batch_scores = [s["score"] for s in batch_scores]
            scores.extend(batch_scores)
        except:
            raise gr.Error(resp.decode())
    documents = [doc for _, doc in sorted(zip(scores, documents))[-k:]]

    return documents