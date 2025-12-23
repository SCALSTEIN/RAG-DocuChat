"""
FastAPI server exposing a /query endpoint.

Run with:
    uvicorn app.server:app --reload --port 8000
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from .query import answer_query

load_dotenv()

app = FastAPI(title="RAG Q&A API")


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4
    conversational: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = None


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question or req.question.strip() == "":
        raise HTTPException(status_code=400, detail="question is required")
    result = answer_query(req.question, k=req.k, conversational=req.conversational)
    sources = []
    for doc in result.get("source_documents", []) or []:
        sources.append({"source": doc.metadata.get("source", "unknown"), "chunk": doc.metadata.get("chunk", None)})
    return {"answer": result["answer"], "sources": sources}