# RAG-DocuChat
This repository demonstrates a Retrieval-Augmented Generation (RAG) Q&A system using Python and LangChain. The system lets users query document datasets naturally — documents are ingested, chunked, embedded into a vector DB (Chroma), and queried with an LLM (OpenAI by default).

Features
- Ingest files (PDF, TXT, MD, CSV) or directories
- Chunking + metadata preservation
- Chroma vector store (persisted)
- Query endpoint (FastAPI) and CLI example
- Supports OpenAI embeddings & chat model by default, with options for HuggingFace-based embeddings/LLMs

- Quickstart
1. Create virtual environment and install
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Set environment variables (see `.env.example`):
   - OPENAI_API_KEY (required for the default path)
   Optionally set:
   - PERSIST_DIRECTORY (where Chroma is persisted), default: `./chroma_db`

3. Ingest documents (example)
   python -m app.ingest --docs ./sample_docs --persist_dir ./chroma_db

4. Run the API
   uvicorn app.server:app --reload --port 8000

5. Query
   - HTTP POST /query with JSON { "question": "Your question" }
   - CLI: python -m app.query --question "What is X?"

Notes and tips
- You can switch embeddings to HuggingFace / sentence-transformers in `rag/ingest.py` by passing `embedding_backend="hf"`.
- For multi-turn chat, use the `conversational=True` option in query functions to use LangChain's conversational retrieval chain.
- Tune chunk_size and chunk_overlap for your dataset (smaller chunks for dense content, larger for context-heavy docs).
- This example uses Chroma for persistence and OpenAI for embeddings and LLM generation. For production consider vector DB options like Pinecone, Weaviate, Milvus, or managed Chroma, and proper secrets & rate-limiting.

Security & Costs
- Be mindful of API usage and cost when using hosted LLMs.
- Protect API keys (do not commit .env with secret keys).

Next steps (ideas)
- Add document ingestion metadata (source, page numbers) and surface citations in answers
- Add re-ranking or hybrid retrieval (BM25 + embedding)
- Add streaming responses and conversation memory
- Add authentication to the API, usage quotas, and monitoring
