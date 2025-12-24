
import streamlit as st
import os
import tempfile
import pickle
import time
import google.generativeai as genai

# Core LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Agent Imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool

# Retrieval Imports
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Tools
from langchain_community.tools import DuckDuckGoSearchRun

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat Agent", page_icon="🕵️", layout="wide")
st.title("🕵️ DocuChat Agent: Safe Mode")

# --- GLOBAL CONSTANTS ---
DB_PATH = "vector_db"
SPLITS_PATH = "splits.pkl"

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Mission Control")
    
    # 1. API Keys
    with st.expander("🔐 API Keys", expanded=True):
        google_api_key = st.text_input("Google API Key", type="password")
    
    # 2. Model Selector
    st.markdown("### 🤖 Agent Brain")
    valid_models = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.0-pro"
    ]
    selected_model = st.selectbox("Select Model:", valid_models, index=0)

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
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def get_embeddings(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def process_documents(files, _api_key):
    if not files and not os.path.exists(DB_PATH):
        return None, None
    
    # Initialize embeddings
    embeddings = get_embeddings(_api_key)

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
            
        # Create smaller chunks to avoid hitting limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        
        # --- SAFE MODE INGESTION ---
        # We process in batches of 5 to avoid 429 Rate Limit
        vector_store = None
        batch_size = 5
        total_batches = len(splits) // batch_size + 1
        
        progress_bar = st.progress(0, text="Embedding documents safely...")
        
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            if not batch:
                continue
                
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            
            # Update Progress
            current_batch = (i // batch_size) + 1
            progress = min(current_batch / total_batches, 1.0)
            progress_bar.progress(progress, text=f"Processing batch {current_batch}/{total_batches}...")
            
            # WAIT to respect rate limit
            time.sleep(2) 
            
        progress_bar.empty()
        
        vector_store.save_local(DB_PATH)
        
        with open(SPLITS_PATH, "wb") as f:
            pickle.dump(splits, f)
            
        return vector_store, splits

    # B. Load Existing
    elif os.path.exists(DB_PATH) and os.path.exists(SPLITS_PATH):
        try:
            vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(SPLITS_PATH, "rb") as f:
                splits = pickle.load(f)
            return vector_store, splits
        except Exception as e:
            st.error(f"Corruption detected: {e}")
            return None, None
            
    return None, None

def build_advanced_retriever(vector_store, splits):
    # 1. Semantic Search (FAISS)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 2. Keyword Search (BM25)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4
    
    # 3. Hybrid Ensemble (50% Semantic / 50% Keyword)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

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

for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
    with st.chat_message(role):
        st.markdown(content)

if google_api_key:
    try:
        llm = get_llm(google_api_key, selected_model)
        
        # Pass API Key for Safe Mode Processing
        vector_store, splits = process_documents(uploaded_files, google_api_key)
        
        if vector_store and splits:
            retriever = build_advanced_retriever(vector_store, splits)
            agent_executor = create_agent(llm, retriever)
            
            if prompt := st.chat_input("Ask about the PDF or the Web..."):
                st.chat_message("user").markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Agent is working..."):
                        response = agent_executor.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.messages
                        })
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                st.session_state.messages.append(HumanMessage(content=prompt))
                st.session_state.messages.append(AIMessage(content=output_text))

        else:
            if not uploaded_files:
                st.info("ℹ️ No PDF uploaded. Switching to Web-Only mode.")
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
    st.warning("Please provide Google API Key to initialize the Agent.")
