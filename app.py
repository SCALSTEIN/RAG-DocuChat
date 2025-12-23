import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# Import HuggingFace for local embeddings (No API limits!)
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat (Free Version)", page_icon="🤖")
st.title("🤖 Chat with PDF (No Rate Limits)")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("[Get a Free Google Key](https://aistudio.google.com/app/apikey)")

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

@st.cache_resource
def process_document(file_path):
    # Notice: We don't need the API key here anymore!
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # USE LOCAL EMBEDDINGS (HuggingFace)
    # This runs on the CPU and does not hit Google's API limit
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, api_key):
    # We still use Google Gemini for the ACTUAL answering
    os.environ["GOOGLE_API_KEY"] = api_key
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

if uploaded_file and api_key: # Check for API key before processing
    try:
        with st.spinner("Processing document... (This runs locally)"):
            temp_path = save_uploaded_file(uploaded_file)
            
            # 1. Create Vector Store (Locally - No Rate Limit)
            vector_store = process_document(temp_path)
            
            # 2. Setup QA Chain (Uses Google API only for answering)
            rag_chain = get_rag_chain(vector_store, api_key)

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
elif uploaded_file and not api_key:
    st.warning("Please enter your Google API Key to chat.")
else:
    st.info("Please provide a Google API Key and upload a PDF to start.")

