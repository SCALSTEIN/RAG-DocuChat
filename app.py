import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat Pro", page_icon="🧠", layout="wide")
st.title("🧠 DocuChat Pro: Memory, Multi-File & Persistence")

# --- GLOBAL CONSTANTS ---
INDEX_path = "faiss_index"

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Keys
    with st.expander("🔐 API Keys", expanded=True):
        google_api_key = st.text_input("Google API Key", type="password")
        hf_token = st.text_input("Hugging Face Token", type="password")
    
    # 2. Dynamic Model Selector
    st.markdown("### 🤖 Model")
    available_models = []
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception:
            pass # Fail silently if key is bad
            
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
    st.markdown("### 📂 Data Source")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚠️ Delete Saved Index"):
        if os.path.exists(INDEX_path):
            import shutil
            shutil.rmtree(INDEX_path)
            st.success("Index deleted. Please reload.")
            st.rerun()

# --- HELPER FUNCTIONS ---

def get_llm(api_key, model_name):
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def get_embeddings(hf_token):
    os.environ['HF_TOKEN'] = hf_token
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_or_create_vector_store(files, _embeddings):
    """
    STRICT RULE: No st.toast or st.write inside this function because it is cached.
    """
    # Case 1: Processing New Files
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
        
        vector_store = FAISS.from_documents(splits, _embeddings)
        vector_store.save_local(INDEX_path)
        return vector_store, "new" # Return tuple to signal what happened
    
    # Case 2: Load Existing Index
    elif os.path.exists(INDEX_path):
        try:
            vector_store = FAISS.load_local(
                INDEX_path, 
                _embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store, "loaded"
        except Exception as e:
            return None, str(e)
            
    return None, None

def get_conversational_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    runnable_filter = contextualize_q_prompt | llm | StrOutputParser()
    
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    final_chain = (
        RunnablePassthrough.assign(
            rephrased_question=runnable_filter
        ) 
        | RunnablePassthrough.assign(
            question=lambda x: x["rephrased_question"] 
        )
        | rag_chain
    )
    
    return final_chain

# --- MAIN APP LOGIC ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History (With Safe Type Checking)
for message in st.session_state.messages:
    if isinstance(message, dict): # Handle legacy dict format
        role = message.get("role", "user")
        content = message.get("content", "")
    else: # Handle new object format
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content

    with st.chat_message(role):
        st.markdown(content)

# Main Execution
if google_api_key and hf_token:
    try:
        embeddings = get_embeddings(hf_token)
        llm = get_llm(google_api_key, selected_model)
        
        # Load Vector Store
        # We now unpack the tuple: (store, status)
        vector_store, status = load_or_create_vector_store(uploaded_files, embeddings)
        
        # Handle UI notifications OUTSIDE the function
        if status == "new":
            st.toast("Index created and saved!", icon="💾")
        elif status == "loaded":
            # Optional: st.toast("Loaded existing index", icon="📂")
            pass
        elif status and "error" in status:
            st.error(f"Error loading index: {status}")

        if vector_store:
            chain = get_conversational_chain(vector_store, llm)
            
            if prompt := st.chat_input("Ask a question..."):
                st.chat_message("user").markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chain.invoke({
                            "question": prompt, 
                            "chat_history": st.session_state.messages
                        })
                        st.markdown(response)
                
                st.session_state.messages.append(HumanMessage(content=prompt))
                st.session_state.messages.append(AIMessage(content=response))
        else:
            if not uploaded_files:
                st.info("👆 Please upload a PDF to create a new index, or ensure a saved index exists.")
                
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please provide both API Keys in the sidebar.")
