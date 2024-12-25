## Ready-to-use containerized RAG system built on top of your knowledge base. Wrapped as a Gradio app.

Components:
- TEI ([text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference))
- [LanceDB](https://lancedb.github.io/lancedb/)/[Qdrant](https://qdrant.tech/documentation/quick-start/)
- OpenAI/HuggingFace-hosted LLM

**!! The input documents are expected to be clean raw text and chunked as needed, they'll be embedded as-is !!**

Usage:

Set `DOCS_DIR` in compose files to the path of your documents directory. 
Then

```
docker compose -f compose.cpu.yaml up
docker compose -f compose.gpu.yaml up
```

GPU configs are currently set to Turing GPUs(T4). Adjust the containers for the `embed` and `rerank` services as necessary ([TEI images](https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images)).

Set it up via modifying `.env`:

```
TOP_K_RETRIEVE - number of items to retrieve before ranking, ignored if a cross-encoder is off
TOP_K_RANK - number of items to use as LLM context. If a cross-encoder is not used this is the number of retrieved items
SEMAPHORE_LIMIT - the limit to the number of coroutines simultaneously working on embedding and ingestion of the documents from DOCS_DIR
BATCH_SIZE - size of batches sent to a TEI embedder on the init stage and TEI reranker during inference. With concurrency and a GPU can be safely set to 1

HF_MODEL - name of a HF LLM used
HF_URL - url to a HF LLM used. Can be a model name if a model is hosted via HuggingChat or a url if you host a model yourself
OPENAI_MODEL - name of an OpenAI LLM
EMBED_MODEL - name of an embedding model from the HF Hub, must match the model used for an embedder

# Indexing and search parameters - parameters of vector DBs, refer to their documentation. [LanceDB](https://blog.lancedb.com/benchmarking-lancedb-92b01032874a), [Qdrant](https://qdrant.tech/documentation/tutorials/retrieval-quality/#tweaking-the-hnsw-parameters)
# Generation parameters - mostly standard [generation params](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig).
PROMPT_TOKEN_LIMIT might come in handy if you expect the context length to go beyond the limit. It'll truncate only context and keep the postfix with appropriate special tokens
REP_PENALTY is a repetition penalty, HF models-specific;
FREQ_PENALTY - frequency penalty, OpenAI models-specific;
```

The secrets `HF_TOKEN` and `OPENAI_API_KEY` must be stored separately in the files with the corresponding names.

**Note that cross-encoder reranking would be very slow on CPU**

-------
Now part of the [Nebius Academy repo](https://github.com/Nebius-Academy/LLMOps-Essentials/tree/rag_service)
