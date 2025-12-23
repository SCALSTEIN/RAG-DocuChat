"""
Query a persisted vector store and return a generated answer.

Usage (CLI):
    python -m app.query --question "What is the support policy?"

This module also exposes helper functions for programmatic use.
"""
import os
import argparse
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
DEFAULT_PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")


def get_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR):
    # Embeddings class here is only needed to construct same embedding type for Chroma.
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db


def answer_query(question: str, persist_dir: str = DEFAULT_PERSIST_DIR, k: int = 4, llm_model: str = "gpt-4o-mini", temperature: float = 0.0, conversational: bool = False, chat_history: Optional[list] = None):
    """
    Run a retrieval + generation pipeline.

    conversational: if True, returns a conversational chain that honors chat_history (a list of (user,assistant) pairs).
    """
    db = get_vectorstore(persist_dir)
    retriever = db.as_retriever(search_kwargs={"k": k})

    # Chat model: default to OpenAI Chat model (adjust model name as needed)
    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    if conversational:
        # ConversationalRetrievalChain keeps history and can return sources
        chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
        result = chain({"question": question, "chat_history": chat_history or []})
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        return {"answer": answer, "source_documents": source_docs}
    else:
        # Simple RetrievalQA chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
        result = chain({"query": question})
        return {"answer": result["result"] if "result" in result else result.get("answer"), "source_documents": result.get("source_documents", [])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--conversational", action="store_true")
    args = parser.parse_args()

    out = answer_query(args.question, persist_dir=args.persist_dir, k=args.k, conversational=args.conversational)
    print("Answer:\n", out["answer"])
    if out.get("source_documents"):
        print("\nSources:")
        for s in out["source_documents"]:
            src = getattr(s, "metadata", {}).get("source", "unknown")
            print(f"- {src} (chunk {s.metadata.get('chunk') if s.metadata else '?'})")