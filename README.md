# 🕵️ DocuChat Agent: Autonomous RAG & Research Assistant

**DocuChat Agent** is a production-grade AI research tool built with **Streamlit** and **LangChain**. It transforms static PDF documents into an interactive knowledge base using advanced retrieval techniques and agentic workflows.

Unlike standard RAG apps, this agent uses **Hybrid Search (BM25 + FAISS)** and **Reranking (Flashrank)** to ensure maximum accuracy, and it can autonomously decide whether to answer from your documents or search the live web.

---

## 🚀 Key Features

### 🧠 **Advanced Retrieval Engine**
* **Hybrid Search:** Combines **Semantic Search** (FAISS) with **Keyword Search** (BM25) to capture both conceptual meaning and exact phrasing.
* **Reranking:** Uses **Flashrank** (a lightweight Cross-Encoder) to re-score the top retrieval results, significantly reducing hallucinations and improving relevance.
* **Local Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` locally on the CPU to avoid API rate limits and costs.

### 🤖 **Agentic Workflow**
* **Autonomous Decision Making:** The AI isn't just a chatbot; it's an **Agent**. It analyzes your question and decides:
    * *"Should I search the PDF?"* (for specific document queries)
    * *"Should I search the Web?"* (for current events, facts, or comparisons)
    * *"Should I use both?"*
* **Web Search Integration:** Powered by **DuckDuckGo** for privacy-focused, real-time internet access.

### 🛠️ **Production-Ready Features**
* **Dynamic Model Selector:** Automatically fetches valid Google Gemini models (e.g., `gemini-1.5-flash`, `gemini-2.0`) available to your specific API key.
* **Persistence:** Automatically saves the Vector Database and text splits to disk. You can close the tab and return later without re-processing your files.
* **Multi-File Support:** Upload and ingest multiple PDFs simultaneously.

---

## 🏗️ Architecture

1.  **Ingestion:** PDFs are loaded via `PyPDFLoader`, split into chunks, and embedded using `HuggingFaceEmbeddings`.
2.  **Indexing:**
    * **Vector Index:** Created using FAISS.
    * **Keyword Index:** Created using BM25.
3.  **Retrieval:** An `EnsembleRetriever` fetches results from both indices (50/50 weight).
4.  **Refinement:** A `ContextualCompressionRetriever` (Flashrank) reranks the top 10 results to find the top 5 true matches.
5.  **Generation:** A LangChain `ToolCallingAgent` orchestrates the retrieval tools and generates the final answer using **Google Gemini**.

---

## 📦 Installation

### Prerequisites
* Python 3.10+
* **Google API Key** (Get it [here](https://aistudio.google.com/app/apikey))
* **HuggingFace Access Token** (Get it [here](https://huggingface.co/settings/tokens))

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/rag-docuchat.git](https://github.com/yourusername/rag-docuchat.git)
    cd rag-docuchat
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 🖥️ Usage

1.  **Configure API Keys:**
    * Open the sidebar.
    * Enter your **Google API Key** (for the Brain).
    * Enter your **HuggingFace Token** (for the Embeddings).
2.  **Select Model:**
    * Wait for the "Model" dropdown to populate. Select a model like `models/gemini-1.5-flash`.
3.  **Upload Data:**
    * Upload one or more PDF documents.
    * Wait for the "Agent is working..." spinner to finish processing.
4.  **Chat:**
    * **Ask about the PDF:** "What are the key financial figures on page 3?"
    * **Ask about the Web:** "What is the current stock price of Apple?"
    * **Ask Complex Questions:** "Compare the budget in this PDF to the 2024 US Federal Budget."

---

## 📁 Project Structure

* `app.py`: Main Streamlit application containing the UI, Agent logic, and RAG pipeline.
* `requirements.txt`: List of Python dependencies.
* `vector_db/`: (Generated) Folder storing the persistent FAISS index.
* `splits.pkl`: (Generated) File storing raw text chunks for BM25 reconstruction.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
