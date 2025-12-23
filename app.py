import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA  # <--- The stable, universal import

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat RAG", page_icon="📄")
st.title("📄 Chat with your PDF")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

@st.cache_resource
def process_document(file_path, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def get_qa_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Use RetrievalQA - works on ALL versions of LangChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True  # This allows us to see the sources
    )
    return qa_chain

# --- MAIN APP LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if api_key and uploaded_file:
    try:
        with st.spinner("Processing document..."):
            temp_path = save_uploaded_file(uploaded_file)
            vector_store = process_document(temp_path, api_key)
            qa_chain = get_qa_chain(vector_store)

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Invoke the chain
                    response = qa_chain.invoke({"query": prompt})
                    
                    answer = response["result"]
                    source_docs = response["source_documents"]

                    st.markdown(answer)

                    with st.expander("Reference Source"):
                        for doc in source_docs:
                            page_num = doc.metadata.get('page', 0) + 1
                            preview = doc.page_content[:100].replace('\n', ' ')
                            st.markdown(f"- **Page {page_num}**: \"{preview}...\"")

            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please provide an OpenAI API Key and upload a PDF to start.")

