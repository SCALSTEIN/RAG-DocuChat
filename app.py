import streamlit as st
import os
import tempfile
import time
from typing import List, Any, Optional

# --- CRITICAL: This MUST be the first Streamlit command ---
st.set_page_config(page_title="DocuChat: Pinecone Enterprise", page_icon="🌲", layout="wide")

# --- PROOF OF SKILL: Production Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# RESUME PROOF: "Develop and maintain vector database infrastructures (Pinecone)"
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# ==============================================================================
# 🏗️ ARCHITECTURE LAYER: Knowledge Base (Pinecone Edition)
# Resume Proof: "Develop and maintain vector database infrastructures (Pinecone)"
# ==============================================================================
class KnowledgeBase:
    """
    Manages the ingestion, chunking, and Cloud Storage (Pinecone).
    Uses Hugging Face for Embeddings + Pinecone for Storage.
    """
    def __init__(self, hf_token: str, pinecone_api_key: str, index_name: str):
        if not hf_token or not pinecone_api_key or not index_name:
            st.error("❌ Missing Keys! Please provide Google, HF, and Pinecone keys.")
            st.stop()
            
        # --- CRITICAL FIX: Set Env Var for LangChain ---
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
            
        # 1. Initialize Embeddings (Hugging Face)
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Initialize Pinecone Client
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        
        # Resume Proof: "Develop and maintain vector database infrastructures"
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.vector_store = None

    def ingest(self, files: List[Any]):
        """
        Ingests files -> Embeds (HF) -> Upserts to Cloud (Pinecone).
        """
        all_docs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.remove(tmp_path)
            
        # Resume Proof: "Prompt optimization / Reranking strategies" (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        
        # --- CLOUD UPLOAD LAYER ---
        with st.status("🌲 Uploading to Pinecone Cloud...", expanded=True) as status:
            try:
                # We use specific namespace to keep demo clean
                namespace = "docuchat_demo"
                
                # Resume Proof: "Large-scale embedding storage" (Batch Processing)
                batch_size = 32
                total_batches = len(splits) // batch_size + 1
                
                for i in range(0, len(splits), batch_size):
                    batch = splits[i : i + batch_size]
                    status.write(f"Upserting batch {i//batch_size + 1}/{total_batches}...")
                    
                    if self.vector_store is None:
                        # First batch creates the connection
                        self.vector_store = PineconeVectorStore.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            index_name=self.index_name,
                            namespace=namespace
                        )
                    else:
                        # Subsequent batches add to it
                        self.vector_store.add_documents(batch)
                    
                    time.sleep(1) # Rate limit protection
                
                status.update(label="✅ Upload Complete!", state="complete", expanded=False)
                return self.vector_store, splits
                
            except Exception as e:
                st.error(f"Pinecone Error: {e}")
                st.stop()

    def connect_existing(self):
        """Connects to an existing Pinecone Index without re-uploading."""
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace="docuchat_demo"
            )
            return self.vector_store
        except Exception:
            return None

# ==============================================================================
# 🧠 INTELLIGENCE LAYER: RAG Pipeline
# Resume Proof: "Architect and implement RAG pipelines, including hybrid search..."
# ==============================================================================
class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def build_retriever(self, k: int = 4):
        # 1. Cloud Dense Retriever (Pinecone)
        pinecone_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # NOTE: For true Hybrid in Pinecone, we'd need 'pinecone-text'. 
        # For this demo, we rely on the high-quality Dense Retrieval of Pinecone
        # combined with the Agent's reasoning capabilities.
        return pinecone_retriever

# ==============================================================================
# 🤖 AGENT LAYER: Orchestration
# ==============================================================================
class AgentEngine:
    def __init__(self, api_key: str, model_name: str):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def create_agent(self, retriever):
        retriever_tool = create_retriever_tool(
            retriever,
            "search_pdf_documents",
            "Use this tool to find information specifically contained in the uploaded PDF documents."
        )
        
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
        1. If the question implies specific internal documents, use 'search_pdf_documents'.
        2. If the question implies real-time data, use 'search_internet'.
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
# ==============================================================================
def main():
    st.title("🌲 DocuChat: Pinecone Enterprise Architecture")
    st.markdown("""
    *Demonstrating Cloud-Native RAG capabilities using Pinecone Vector Database.*
    """)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔐 Cloud Credentials", expanded=True):
            google_api_key = st.text_input("Google API Key", type="password")
            hf_token = st.text_input("Hugging Face Token", type="password")
            pinecone_key = st.text_input("Pinecone API Key", type="password")
            index_name = st.text_input("Index Name", value="docuchat")
        
        st.subheader("🧠 Model Config")
        model_name = st.selectbox("LLM Model", ["models/gemini-1.5-flash", "models/gemini-1.5-pro"])
        
        st.subheader("📂 Ingestion Engine")
        uploaded_files = st.file_uploader("Data Source (PDF)", type="pdf", accept_multiple_files=True)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT HISTORY ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg.role).markdown(msg.content)

    # --- MAIN EXECUTION ---
    if google_api_key and hf_token and pinecone_key:
        try:
            # 1. Init Layers
            kb = KnowledgeBase(hf_token, pinecone_key, index_name)
            
            # 2. Ingest or Connect
            vector_store = None
            if uploaded_files:
                vector_store, _ = kb.ingest(uploaded_files)
            else:
                # Try connecting to existing index if no new files
                vector_store = kb.connect_existing()
            
            # 3. Build Pipeline
            if vector_store:
                rag_pipeline = RAGPipeline(vector_store)
                retriever = rag_pipeline.build_retriever()
                
                engine = AgentEngine(google_api_key, model_name)
                agent_executor = engine.create_agent(retriever)
                
                # 4. Chat Interface
                if prompt := st.chat_input("Ask about the docs or the web..."):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🌲 Pinecone is retrieving..."):
                            response = agent_executor.invoke({
                                "input": prompt,
                                "chat_history": st.session_state.messages
                            })
                            output = response["output"]
                            st.markdown(output)
                                
                    st.session_state.messages.append(AIMessage(content=output))
            else:
                 if not uploaded_files:
                    st.info("👆 Please upload a PDF to upload vectors to Pinecone.")

        except Exception as e:
            st.error(f"System Error: {e}")
    else:
        st.warning("Please provide ALL API Keys (Google, HF, Pinecone) to initialize.")

if __name__ == "__main__":
    main()
