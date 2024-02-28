import asyncio
import time
import os
import logging

from qdrant_client import models, AsyncQdrantClient
from qdrant_client.models import PointStruct
from aiohttp import ClientSession
from pathlib import Path
from transformers import AutoConfig
from tqdm.asyncio import tqdm_asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = AutoConfig.from_pretrained(os.getenv("EMBED_MODEL"))

TEI_URL = os.getenv("EMBED_URL") + "/embed"
DIRPATH = "/usr/src/docs_dir"
TABLE_NAME = "docs"
EMB_DIM = config.hidden_size
M=int(os.getenv("M"))
EF_CONSTRUCT=int(os.getenv("EF_CONSTRUCT"))

q_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), prefer_grpc=True)

HEADERS = {
    "Content-Type": "application/json"
}


async def request(sentence: str, id: int | str, semaphore: asyncio.BoundedSemaphore):
    """
    Vectorize and ingest a single sentence
    Args:
        sentence: a sentence to add to a DB
        id: a unique ID of a record
        semaphore: a semaphore to bound the number of coroutines
    """
    async with semaphore:
        payload = {
            "inputs": sentence,
            "truncate": True
        }

        async with ClientSession(headers=HEADERS) as session:
            async with session.post(TEI_URL, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(await resp.text())
                result = await resp.json()
            
            await q_client.upsert(
                collection_name=TABLE_NAME,
                points=[
                    PointStruct(
                        id=id,
                        vector=result[0],
                        payload={
                            "text": sentence
                        }
                    )
                ],
            )


async def main(texts):
    await q_client.recreate_collection(
        collection_name=TABLE_NAME,
        vectors_config=models.VectorParams(
            size=EMB_DIM,
            distance=models.Distance.EUCLID,
            on_disk=True
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )
    semaphore = asyncio.BoundedSemaphore(int(os.getenv("SEMAPHORE_LIMIT")))
    jobs = [asyncio.ensure_future(request(t, i, semaphore)) for i, t in enumerate(texts)]
    print(f"num jobs: {len(jobs)}")

    await tqdm_asyncio.gather(*jobs)
    await q_client.update_collection(
        collection_name=TABLE_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        hnsw_config=models.HnswConfigDiff(
            m=M,
            ef_construct=EF_CONSTRUCT,
        ),
    )


if __name__ == "__main__":
    start = time.time()
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

    asyncio.run(main(texts))
    logger.info(f"Embedding and ingestion of {len(texts)} items took {time.time() - start}")
