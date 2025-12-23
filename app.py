import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat (Free Version)", page_icon="🤖")
st.title("🤖 Chat with PDF (Authenticated)")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    
    st.markdown("### 1. API Keys")
    google_api_key = st.text_input("Google API Key", type="password")
    
    st.markdown("### 2. Hugging Face Token")
    st.caption("Required to bypass Streamlit rate limits.")
    hf_token = st.text_input("HF Token", type="password")
    st.markdown("[Get Free HF Token](https://huggingface.co/settings/tokens)")
    
    st.markdown("### 3. Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

@st.cache_resource
def process_document(file_path, hf_token):
    # Set HF Token to bypass rate limits
    os.environ['HF_TOKEN'] = hf_token
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # LOCAL EMBEDDINGS (Authenticated)
    # Using a different model alias that is often more reliable
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, google_api_key):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Using 'gemini-1.5-flash' as it is the most standard current model.
    # If 1.5 fails, try 'gemini-pro'
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
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

# Only run if we have BOTH keys and the file
if uploaded_file and google_api_key and hf_token:
    try:
        with st.spinner("Processing document..."):
            temp_path = save_uploaded_file(uploaded_file)
            
            # Pass the HF Token to the processor
            vector_store = process_document(temp_path, hf_token)
            
            rag_chain = get_rag_chain(vector_store, google_api_key)

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
    # Helper warnings if user forgets keys
    if not google_api_key:
        st.warning("⚠️ Please enter your Google API Key.")
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face Token.")
else:
    st.info("Please provide API keys and upload a PDF to start.")
