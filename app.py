import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    """Save uploaded file to a temp directory."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

@st.cache_resource
def process_document(file_path, api_key):
    """Load, split, and embed the document."""
    os.environ["OPENAI_API_KEY"] = api_key
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def get_retrieval_chain(vector_store):
    """
    Build the Chain that returns both the answer and the sources.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever()

    # 1. Define the "Answer" prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based ONLY on the following context. 
    If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {input}
    """)

    # 2. Create the "Document Chain" (Generates the answer)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3. Create the "Retrieval Chain" (Combines Retriever + Document Chain)
    # This automatically passes retrieved docs into the 'context' variable
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- MAIN APP LOGIC ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources in history, display them (optional simplification)
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **Page {source['page']}**: {source['content']}...")

if api_key and uploaded_file:
    try:
        with st.spinner("Processing document..."):
            temp_path = save_uploaded_file(uploaded_file)
            vector_store = process_document(temp_path, api_key)
            chain = get_retrieval_chain(vector_store)

        if prompt := st.chat_input("Ask a question..."):
            
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Generate and Display Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # 'input' is the key expected by create_retrieval_chain
                    response = chain.invoke({"input": prompt})
                    
                    answer = response["answer"]
                    source_docs = response["context"]

                    st.markdown(answer)

                    # 3. Format and Display Sources
                    sources_data = []
                    with st.expander("Reference Source"):
                        for doc in source_docs:
                            # Extract page number (PyPDFLoader uses 0-indexing, so +1)
                            page_num = doc.metadata.get('page', 0) + 1
                            # Preview text (first 100 chars)
                            preview = doc.page_content[:100].replace('\n', ' ')
                            
                            st.markdown(f"- **Page {page_num}**: \"{preview}...\"")
                            sources_data.append({"page": page_num, "content": preview})

            # 4. Save Assistant Message with Sources to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources_data
            })

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please provide an OpenAI API Key and upload a PDF to start.")
