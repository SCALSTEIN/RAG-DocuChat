import streamlit as st
import os
import tempfile
import pickle
import time
# --- FIX: Added List and Any here ---
from typing import List, Any, Optional 

# --- PROOF OF SKILL: Production Libraries ---
from langchain_community.document_loaders import PyPDFLoader

class KnowledgeBase:
    """
    Manages the ingestion, chunking, and storage of documents.
    Uses Hugging Face Inference API with Retry Logic for resilience.
    """
    def __init__(self, hf_token: str):
        if not hf_token:
            st.error("❌ Hugging Face Token is missing! Embeddings cannot be generated.")
            st.stop()
            
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def ingest(self, files: List[Any], chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Ingests files using robust chunking strategies and retry logic.
        """
        all_docs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.remove(tmp_path)
            
        # Resume Proof: "RAG Pipelines... prompt optimization" (Chunking strategies)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(all_docs)
        
        # --- RESILIENCY LAYER: Retry Logic ---
        # Resume Proof: "Ensure reliability in production"
        max_retries = 3
        
        with st.status("🏗️ Ingesting knowledge base...", expanded=True) as status:
            for attempt in range(max_retries):
                try:
                    status.write(f"Attempt {attempt+1}/{max_retries}: Sending data to Hugging Face Inference API...")
                    self.vector_store = FAISS.from_documents(splits, self.embeddings)
                    status.update(label="✅ Embeddings Generated Successfully!", state="complete", expanded=False)
                    break # Success! Exit loop.
                except Exception as e:
                    if attempt < max_retries - 1:
                        status.write(f"⚠️ API Error (Likely Cold Start). Waiting 5s for model to wake up...")
                        time.sleep(5) # Wait for model to load on HF server
                    else:
                        status.update(label="❌ Embedding Failed", state="error")
                        st.error(f"Final Error from Hugging Face: {e}")
                        st.info("Troubleshooting: Check your HF Token, or try again in 1 minute (API might be overloaded).")
                        st.stop()
        
        # Persistence
        self.vector_store.save_local(DB_PATH)
        with open(SPLITS_PATH, "wb") as f:
            pickle.dump(splits, f)
            
        return self.vector_store, splits

