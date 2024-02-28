import requests
import time
import numpy as np
import pyarrow as pa
import lancedb
import logging
import os

from tqdm import tqdm
from pathlib import Path
from transformers import AutoConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEI_URL= os.getenv("EMBED_URL") + "/embed"
DIRPATH = "/usr/src/docs_dir"
TABLE_NAME = "docs"
config = AutoConfig.from_pretrained(os.getenv("EMBED_MODEL"))
EMB_DIM = config.hidden_size
CREATE_INDEX = int(os.getenv("CREATE_INDEX"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS"))
NUM_SUB_VECTORS = int(os.getenv("NUM_SUB_VECTORS"))

HEADERS = {
    "Content-Type": "application/json"
}


def embed_and_index():
    files = Path(DIRPATH).rglob("*")
    texts = []

    for file in files:
        if file.is_file():
            try:
                text = file.open().read()
                if text:
                    texts.append(text)
            except (OSError, UnicodeDecodeError) as e:
                logger.error("Error reading file: ", e)
            except Exception as e:
                logger.error("Unhandled exception: ", e)
                raise

    logger.info(f"Successfully read {len(texts)} files")

    db = lancedb.connect("/usr/src/.lancedb")
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), EMB_DIM)),
            pa.field("text", pa.string()),
        ]
    )
    tbl = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")

    start = time.time()
    
    for i in tqdm(range(int(np.ceil(len(texts) / BATCH_SIZE)))):
        payload = {
            "inputs": texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
            "truncate": True
        }

        resp = requests.post(TEI_URL, json=payload, headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(resp.text)
        vectors = resp.json()

        data = [
            {"vector": vec, "text": text}
            for vec, text in zip(vectors, texts[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        ]
        tbl.add(data=data)
    
    logger.info(f"Embedding and ingestion of {len(texts)} items took {time.time() - start}")

    # IVF-PQ indexing
    if CREATE_INDEX:
        tbl.create_index(num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)


if __name__ == "__main__":
    embed_and_index()