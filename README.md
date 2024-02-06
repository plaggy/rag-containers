## Ready-to-use containerized RAG system built on top of your knowledge base. Wrapped as a Gradio app.

Components:
- TEI ([text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference))
- [LanceDB](https://lancedb.github.io/lancedb/)/[Qdrant](https://qdrant.tech/documentation/quick-start/)
- OpenAI/HuggingFace-hosted LLM


Usage:

```
docker compose -f compose.cpu.yaml up
docker compose -f compose.gpu.yaml up
```

GPU configs are currently set to with Turing GPUs(T4). Adjust the containers for the `embed` and `rerank` services as necessary ([TEI images](https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images)).

Set it up via modifying `.env`:

```
TOP_K_RETRIEVE - number of items to retrieve before ranking, ignored if a cross-encoder is off
TOP_K_RANK - number of items to use as LLM context. If a cross-encoder is not used this is the number of retrieved items
SEMAPHORE_LIMIT - the limit to the number of coroutines simultaneously working on embedding and ingestion of the documents from DOCS_DIR
BATCH_SIZE - size of batches sent to a TEI embedder on the init stage and TEI reranker during inference

HF_MODEL - name of a HF LLM used
HF_URL - url to a HF LLM used. Can be a model name if a model is hosted via HuggingChat or a url if you host a model yourself
OPENAI_MODEL - name of an OpenAI LLM
EMBED_MODEL - name of an embedding model from the HF Hub, must match the model used for an embedder

# Indexing and search parameters - parameters of vector DBs, refer to their documentation
# Generation parameters - standard generation params. REP_PENALTY is a repetition penalty, HF models-specific;
FREQ_PENALTY - frequency penalty, OpenAI models-specific
```

The secrets `HF_TOKEN` and `OPENAI_API_KEY` must be stored separately in the files with the corresponding names.
