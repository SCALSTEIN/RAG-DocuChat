import streamlit as st
import os
import tempfile
import time
from typing import List, Any

# --- CRITICAL: This MUST be the first Streamlit command ---
st.set_page_config(page_title="DocuChat: Pinecone Enterprise", page_icon="🌲", layout="wide")

# --- PROOF OF SKILL: Production Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# RESUME PROOF: "Hybrid Compute" (Local Embedding + Cloud Storage)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings 
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun

# ==============================================================================
# 🏗️ ARCHITECTURE LAYER: Knowledge Base (Hybrid Pinecone)
# Resume Proof: "Develop and maintain vector database infrastructures"
# ==============================================================================
class KnowledgeBase:
    """
    Manages ingestion. 
    Architecture: Local Compute (Embeddings) -> Cloud Storage (Pinecone).
    """
    def __init__(self, pinecone_api_key: str, index_name: str):
        if not pinecone_api_key or not index_name:
            st.error("❌ Missing Pinecone API Key or Index Name.")
            st.stop()
            
        # --- CRITICAL FIX: Set Env Var for LangChain ---
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
            
        # 1. Initialize Embeddings (Local FastEmbed - No API Rate Limits)
        # Uses BAAI/bge-small-en-v1.5 (Default) -> 384 Dimensions
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Initialize Pinecone Client
        self.index_name = index_name
        self.vector_store = None

    def ingest(self, files: List[Any]):
        """
        Ingests files -> Embeds Locally -> Upserts to Cloud.
        """
        all_docs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.remove(tmp_path)
            
        # Resume Proof: "Chunking strategies"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        
        # --- CLOUD UPLOAD LAYER ---
        with st.status("🌲 Processing & Uploading...", expanded=True) as status:
            try:
                namespace = "docuchat_demo"
                # Batching is still good practice even with local embeddings
                batch_size = 32
                total_batches = len(splits) // batch_size + 1
                
                for i in range(0, len(splits), batch_size):
                    batch = splits[i : i + batch_size]
                    status.write(f"Processing batch {i//batch_size + 1}/{total_batches} (Local Embedding -> Pinecone)...")
                    
                    if self.vector_store is None:
                        self.vector_store = PineconeVectorStore.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            index_name=self.index_name,
                            namespace=namespace
                        )
                    else:
                        self.vector_store.add_documents(batch)
                
                status.update(label="✅ Upload Complete!", state="complete", expanded=False)
                return self.vector_store, splits
                
            except Exception as e:
                st.error(f"Pinecone Error: {e}")
                st.info("Tip: Ensure your Pinecone Index dimensions are set to 384.")
                st.stop()

    def connect_existing(self):
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
# ==============================================================================
class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def build_retriever(self, k: int = 4):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# ==============================================================================
# 🤖 AGENT LAYER: Orchestration
# ==============================================================================
class AgentEngine:
    def __init__(self, api_key: str, model_name: str):
        os.environ["GOOGLE_API_KEY"] = api_key
        # FIX: Removed 'models/' prefix which causes 404 errors in some regions/versions
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
        Answer the user's question using the most reliable source.
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
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔐 Cloud Credentials", expanded=True):
            google_api_key = st.text_input("Google API Key", type="password")
            pinecone_key = st.text_input("Pinecone API Key", type="password")
            index_name = st.text_input("Index Name", value="docuchat")
        
        st.subheader("🧠 Model Config")
        # FIX: Updated model list to remove 'models/' prefix
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        selected_model = st.selectbox("LLM Model", model_options)
        
        # Fallback for manual entry
        if st.checkbox("Use Custom Model Name"):
            selected_model = st.text_input("Enter Model Name", "gemini-1.5-flash")
        
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
    if google_api_key and pinecone_key:
        try:
            # 1. Init Layers
            kb = KnowledgeBase(pinecone_key, index_name)
            
            # 2. Ingest or Connect
            vector_store = None
            if uploaded_files:
                vector_store, _ = kb.ingest(uploaded_files)
            else:
                vector_store = kb.connect_existing()
            
            # 3. Build Pipeline
            if vector_store:
                rag_pipeline = RAGPipeline(vector_store)
                retriever = rag_pipeline.build_retriever()
                
                engine = AgentEngine(google_api_key, selected_model)
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
        st.warning("Please provide Google and Pinecone API Keys.")

if __name__ == "__main__":
    main()
