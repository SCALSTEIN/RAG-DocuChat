class KnowledgeBase:
    """
    Manages the ingestion, chunking, and storage of documents.
    Uses Hugging Face Inference API with Batching and Retry Logic for resilience.
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
        Ingests files using robust chunking strategies, batching, and retry logic.
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
        
        # --- OPTIMIZATION LAYER: Batch Processing ---
        # Resume Proof: "Develop and maintain vector database infrastructures for large-scale embedding"
        batch_size = 32  # Small batch size to prevent API Timeouts
        total_batches = len(splits) // batch_size + 1
        
        progress_bar = st.progress(0, text="🏗️ Initializing ingestion...")
        
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            if not batch: continue
            
            # Retry Logic per Batch
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    if self.vector_store is None:
                        self.vector_store = FAISS.from_documents(batch, self.embeddings)
                    else:
                        self.vector_store.add_documents(batch)
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2) # Short wait for API to cool down
                    else:
                        st.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            
            if not success:
                st.error("❌ Critical Failure: Could not embed document even with retries.")
                st.stop()
            
            # Update Progress
            current_progress = min((i // batch_size + 1) / total_batches, 1.0)
            progress_bar.progress(current_progress, text=f"Embedding batch {i//batch_size + 1}/{total_batches}...")
            time.sleep(0.5) # Politeness delay
            
        progress_bar.empty()
        
        # Persistence
        if self.vector_store:
            self.vector_store.save_local(DB_PATH)
            with open(SPLITS_PATH, "wb") as f:
                pickle.dump(splits, f)
            
        return self.vector_store, splits
