import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat (Dynamic)", page_icon="🤖")
st.title("🤖 Chat with PDF (Dynamic Model Fix)")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    
    st.markdown("### 1. API Keys")
    google_api_key = st.text_input("Google API Key", type="password")
    
    # --- DYNAMIC MODEL SELECTOR ---
    st.markdown("### 2. Select Model")
    available_models = []
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            # Fetch all models that support 'generateContent'
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            st.error(f"Could not list models: {e}")
            
    # If we found models, show a dropdown. If not, show a text box as fallback.
    if available_models:
        # Default to the first 'flash' model if available, otherwise the first one
        default_index = 0
        for i, m in enumerate(available_models):
            if "flash" in m:
                default_index = i
                break
        selected_model = st.selectbox("Choose a valid model:", available_models, index=default_index)
    else:
        selected_model = st.text_input("Manually type model name:", "models/gemini-1.5-flash")

    st.markdown("### 3. Hugging Face Token")
    hf_token = st.text_input("HF Token", type="password")
    
    st.markdown("### 4. Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

@st.cache_resource
def process_document(file_path, hf_token):
    os.environ['HF_TOKEN'] = hf_token
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # LOCAL EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, google_api_key, model_name):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Use the EXACT name selected from the dropdown
    # We strip 'models/' prefix if LangChain adds it automatically, 
    # but usually passing the full string 'models/gemini-...' is safest.
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    retriever = vector_store.as_retriever()
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- MAIN APP LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file and google_api_key and hf_token and selected_model:
    try:
        with st.spinner("Processing document..."):
            temp_path = save_uploaded_file(uploaded_file)
            vector_store = process_document(temp_path, hf_token)
            # Pass the selected model to the chain
            rag_chain = get_rag_chain(vector_store, google_api_key, selected_model)

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif uploaded_file:
    if not google_api_key:
        st.warning("⚠️ Please enter your Google API Key.")
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face Token.")
else:
    st.info("Please provide API keys and upload a PDF to start.")
