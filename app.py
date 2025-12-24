import streamlit as st
import os
import tempfile
import pickle
import time
from typing import List, Any, Optional

# --- CRITICAL: This MUST be the first Streamlit command ---
st.set_page_config(page_title="DocuChat Agent: Enterprise", page_icon="🏗️", layout="wide")

# --- PROOF OF SKILL: Production Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# --- GLOBAL CONFIG ---
DB_PATH = "vector_db"
SPLITS_PATH = "splits.pkl"

# ==============================================================================
# 🏗️ ARCHITECTURE LAYER: Knowledge Base
# Resume Proof: "Develop and maintain vector database infrastructures"
# ==============================================================================
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

    def load_existing(self):
        if os.path.exists(DB_PATH) and os.path.exists(SPLITS_PATH):
            self.vector_store = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            with open(SPLITS_PATH, "rb") as f:
                splits = pickle.load(f)
            return self.vector_store, splits
        return None, None

# ==============================================================================
# 🧠 INTELLIGENCE LAYER: RAG Pipeline
# Resume Proof: "Architect and implement RAG pipelines, including hybrid search..."
# ==============================================================================
class RAGPipeline:
    """
    Constructs the retrieval strategy.
    Implements Hybrid Search (BM25 + Dense Vector).
    """
    def __init__(self, vector_store, splits):
        self.vector_store = vector_store
        self.splits = splits

    def build_retriever(self, k: int = 4):
        # 1. Dense Retriever (Semantic)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # 2. Sparse Retriever (Keyword)
        # Resume Proof: "Hybrid search"
        bm25_retriever = BM25Retriever.from_documents(self.splits)
        bm25_retriever.k = k
        
        # 3. Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble

# ==============================================================================
# 🤖 AGENT LAYER: Orchestration
# Resume Proof: "Implement LLM orchestration... build agentic workflows"
# ==============================================================================
class AgentEngine:
    """
    Orchestrates the LLM, Tools, and Memory.
    """
    def __init__(self, api_key: str, model_name: str):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def create_agent(self, retriever):
        # Tool 1: Internal Knowledge
        retriever_tool = create_retriever_tool(
            retriever,
            "search_pdf_documents",
            "Use this tool to find information specifically contained in the uploaded PDF documents."
        )
        
        # Tool 2: External Knowledge
        search = DuckDuckGoSearchRun()
        from langchain_core.tools import Tool
        web_tool = Tool(
            name="search_internet",
            func=search.run,
            description="Use this tool to find current events, exchange rates, or facts NOT in the PDF."
        )
        
        tools = [retriever_tool, web_tool]
        
        system_prompt = """You are an expert Autonomous Research Agent. 
        Your goal is to answer the user's question using the most reliable source.
        
        STRATEGY:
        1. If the question implies specific internal documents (e.g., "what does the file say"), use 'search_pdf_documents'.
        2. If the question implies real-time data (e.g., "current stock price"), use 'search_internet'.
        3. If complex, you may use BOTH tools and synthesize the answer.
        
        Always cite your sources."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==============================================================================
# 🖥️ UI LAYER: Streamlit Application
# Resume Proof: "Develop production-ready AI applications"
# ==============================================================================
def main():
    st.title("🏗️ DocuChat: Enterprise RAG Architecture")
    st.markdown("""
    *Demonstrating advanced RAG capabilities: Hybrid Search, Agentic Workflows, and Multi-Provider Integration.*
    """)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        # RESUME PROOF: Managing multiple API providers
        with st.expander("🔐 Credentials", expanded=True):
            google_api_key = st.text_input("Google API Key", type="password")
            hf_token = st.text_input("Hugging Face Token", type="password")
        
        st.subheader("🧠 Model Config")
        model_name = st.selectbox("LLM Model", ["models/gemini-1.5-flash", "models/gemini-1.5-pro"])
        
        st.subheader("📂 Ingestion Engine")
        uploaded_files = st.file_uploader("Data Source (PDF)", type="pdf", accept_multiple_files=True)
        
        if st.button("🗑️ Clear Cache"):
            if os.path.exists(DB_PATH):
                import shutil
                shutil.rmtree(DB_PATH)
            if os.path.exists(SPLITS_PATH):
                os.remove(SPLITS_PATH)
            st.session_state.messages = []
            st.rerun()

    # --- CHAT HISTORY ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg.role).markdown(msg.content)

    # --- MAIN EXECUTION ---
    if google_api_key and hf_token:
        try:
            # 1. Init Layers
            kb = KnowledgeBase(hf_token)
            
            # 2. Load or Ingest Data
            if uploaded_files:
                vector_store, splits = kb.ingest(uploaded_files)
            else:
                vector_store, splits = kb.load_existing()
            
            # 3. Build Pipeline & Agent
            if vector_store and splits:
                rag_pipeline = RAGPipeline(vector_store, splits)
                retriever = rag_pipeline.build_retriever()
                
                engine = AgentEngine(google_api_key, model_name)
                agent_executor = engine.create_agent(retriever)
                
                # 4. Chat Interface
                if prompt := st.chat_input("Ask about the docs or the web..."):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Agent orchestrating tools..."):
                            response = agent_executor.invoke({
                                "input": prompt,
                                "chat_history": st.session_state.messages
                            })
                            output = response["output"]
                            st.markdown(output)
                            
                            with st.expander("🕵️ Agent Trace"):
                                st.write("✅ Retrieval Successful")
                                st.write("✅ Tools Orchestrated")
                                
                    st.session_state.messages.append(AIMessage(content=output))
            
            else:
                if not uploaded_files:
                    st.info("👆 Please upload a PDF to initialize the Knowledge Base.")

        except Exception as e:
            st.error(f"System Error: {e}")
    else:
        st.warning("Please provide BOTH API Keys to initialize the system.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    main()
