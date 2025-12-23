"""
Ingest documents into a Chroma vector store.

Usage:
    python -m app.ingest --docs ./my_docs --persist_dir ./chroma_db
"""
import os
import argparse
from typing import List, Optional

from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")


def load_files(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        if os.path.isdir(p):
            # DirectoryLoader will try many loader types
            loader = DirectoryLoader(p, loader_cls=TextLoader, recursive=True)
            docs.extend(loader.load())
        else:
            ext = os.path.splitext(p)[1].lower()
            if ext in [".txt", ".md"]:
                loader = TextLoader(p, encoding="utf-8")
                docs.extend(loader.load())
            elif ext in [".pdf"]:
                loader = UnstructuredPDFLoader(p)
                docs.extend(loader.load())
            elif ext in [".csv"]:
                loader = CSVLoader(p)
                docs.extend(loader.load())
            else:
                # Fallback to TextLoader; you can add more loaders
                try:
                    loader = TextLoader(p, encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Skipped {p}: {e}")
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_docs = []
    for d in docs:
        # preserve metadata like source or filename if provided
        pieces = splitter.split_text(d.page_content)
        for i, piece in enumerate(pieces):
            meta = dict(d.metadata) if d.metadata else {}
            meta.setdefault("source", meta.get("source", d.metadata.get("source") if d.metadata else "unknown"))
            meta["chunk"] = i
            new_docs.append(Document(page_content=piece, metadata=meta))
    return new_docs


def create_vectorstore(docs: List[Document], persist_dir: str = DEFAULT_PERSIST_DIR, embedding_backend: str = "openai"):
    if embedding_backend == "openai":
        emb = OpenAIEmbeddings()
    elif embedding_backend == "hf":
        # Example: use sentence-transformers model
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Unknown embedding backend")

    vectordb = Chroma.from_documents(docs, embedding=emb, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


def ingest(paths: List[str], persist_dir: str = DEFAULT_PERSIST_DIR, chunk_size: int = 1000, chunk_overlap: int = 200, embedding_backend: str = "openai"):
    print("Loading files...")
    docs = load_files(paths)
    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} chunks. Creating embeddings and storing into Chroma at {persist_dir} ...")
    db = create_vectorstore(chunks, persist_dir=persist_dir, embedding_backend=embedding_backend)
    print("Ingestion complete.")
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", required=True, help="Paths to files or directories to ingest")
    parser.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", 1000)))
    parser.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", 200)))
    parser.add_argument("--embedding_backend", choices=["openai", "hf"], default="openai")
    args = parser.parse_args()

    ingest(args.docs, persist_dir=args.persist_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, embedding_backend=args.embedding_backend)