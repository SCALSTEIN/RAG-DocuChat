import streamlit as st
import os
import tempfile
import pickle
import google.generativeai as genai

# Core LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool

# Level 1 & 2: Hybrid Search & Reranking
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# Level 3: Tools
from langchain_community.tools import DuckDuckGoSearchRun

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat Agent", page_icon="🕵️", layout="wide")
st.title("🕵️ DocuChat Agent: Hybrid, Reranked & Autonomous")

# --- GLOBAL CONSTANTS ---
DB_PATH = "vector_db"
SPLITS_PATH = "splits.pkl" # We need to save raw text for BM25

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Mission Control")
    
    # 1. API Keys
    with st.expander("🔐 API Keys", expanded=True):
        google_api_key = st.text_input("Google API Key", type="password")
        hf_token = st.text_input("Hugging Face Token", type="password")
    
    # 2. Dynamic Model Selector
    st.markdown("### 🤖 Agent Brain")
    available_models = []
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception:
            pass 
            
    if available_models:
        default_index = 0
        for i, m in enumerate(available_models):
            if "flash" in m:
                default_index = i
                break
        selected_model = st.selectbox("Select Model:", available_models, index=default_index)
    else:
        selected_model = st.text_input("Model Name:", "models/gemini-1.5-flash")

    # 3. Data Management
    st.markdown("### 📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if st.button("🗑️ Reset Agent Memory"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚠️ Delete Knowledge Base"):
        if os.path.exists(DB_PATH):
            import shutil
            shutil.rmtree(DB_PATH)
        if os.path.exists(SPLITS_PATH):
            os.remove(SPLITS_PATH)
        st.success("Deleted. Reloading...")
        st.rerun()

# --- HELPER FUNCTIONS ---

def get_llm(api_key, model_name):
    os.environ["GOOGLE_API_KEY"] = api_key
    # Agent requires tool binding, low temp for precision
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def get_embeddings(hf_token):
    os.environ['HF_TOKEN'] = hf_token
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def process_documents(files, _embeddings):
    """
    1. Reads PDF
    2. Splits Text
    3. Creates FAISS (Vector) Index
    4. Saves FAISS + Raw Splits (for BM25)
    """
    if not files and not os.path.exists(DB_PATH):
        return None, None

    # A. Process New Files
    if files:
        all_docs = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
            os.remove(tmp_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # Save FAISS
        vector_store = FAISS.from_documents(splits, _embeddings)
        vector_store.save_local(DB_PATH)
        
        # Save Splits (Required for BM25 reconstruction)
        with open(SPLITS_PATH, "wb") as f:
            pickle.dump(splits, f)
            
        return vector_store, splits

    # B. Load Existing
    elif os.path.exists(DB_PATH) and os.path.exists(SPLITS_PATH):
        try:
            vector_store = FAISS.load_local(DB_PATH, _embeddings, allow_dangerous_deserialization=True)
            with open(SPLITS_PATH, "rb") as f:
                splits = pickle.load(f)
            return vector_store, splits
        except Exception as e:
            st.error(f"Corruption detected: {e}")
            return None, None
            
    return None, None

def build_advanced_retriever(vector_store, splits):
    # 1. Semantic Search (FAISS)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 2. Keyword Search (BM25) - Level 2
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    
    # 3. Hybrid Ensemble (50% Semantic / 50% Keyword)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. Reranking (Flashrank) - Level 1
    # Compresses top 10 results down to top 4 most relevant
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def create_agent(llm, retriever):
    # Tool 1: The Super-Retriever
    retriever_tool = create_retriever_tool(
        retriever,
        "search_pdf_documents",
        "Search for information inside the uploaded PDF documents. Always use this first for specific questions."
    )
    
    # Tool 2: Web Search
    search = DuckDuckGoSearchRun()
    from langchain_core.tools import Tool
    web_tool = Tool(
        name="search_internet",
        func=search.run,
        description="Useful for finding current events, facts not in the PDF, or general knowledge."
    )
    
    tools = [retriever_tool, web_tool]
    
    # Agent Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. You have access to PDF documents and the Internet. "
                   "Always prefer the PDF documents for specific questions. "
                   "If the PDF doesn't have the answer, try the internet. "
                   "Cite your sources."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- MAIN APP LOGIC ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
    with st.chat_message(role):
        st.markdown(content)

if google_api_key and hf_token:
    try:
        embeddings = get_embeddings(hf_token)
        llm = get_llm(google_api_key, selected_model)
        
        # 1. Load Data
        vector_store, splits = process_documents(uploaded_files, embeddings)
        
        # 2. Build Tools (Only if data exists)
        if vector_store and splits:
            # Build the "Super Retriever"
            retriever = build_advanced_retriever(vector_store, splits)
            
            # Create Agent
            agent_executor = create_agent(llm, retriever)
            
            # User Input
            if prompt := st.chat_input("Ask about the PDF or the Web..."):
                st.chat_message("user").markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Agent is working... (Searching PDF & Web)"):
                        # Agent Execution
                        response = agent_executor.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.messages
                        })
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                # Save History
                st.session_state.messages.append(HumanMessage(content=prompt))
                st.session_state.messages.append(AIMessage(content=output_text))

        else:
            # Fallback if no PDF: Agent with ONLY Web Search
            if not uploaded_files:
                st.info("ℹ️ No PDF uploaded. Switching to Web-Only mode.")
                
                # Simple Web Agent
                search = DuckDuckGoSearchRun()
                from langchain_core.tools import Tool
                web_tool = Tool(
                    name="search_internet",
                    func=search.run,
                    description="Search the web."
                )
                agent = create_tool_calling_agent(llm, [web_tool], ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful web assistant."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]))
                web_agent = AgentExecutor(agent=agent, tools=[web_tool])
                
                if prompt := st.chat_input("Ask the web..."):
                    st.chat_message("user").markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Searching web..."):
                            response = web_agent.invoke({"input": prompt, "chat_history": st.session_state.messages})
                            st.markdown(response["output"])
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    st.session_state.messages.append(AIMessage(content=response["output"]))

    except Exception as e:
        st.error(f"System Error: {e}")

else:
    st.warning("Please provide API Keys to initialize the Agent.")


