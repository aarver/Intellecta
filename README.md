# Intellecta

**Intellecta** is an intelligent question-answering tool that combines information from a webpage with live web search.  
Simply provide a URL, ask a question, and get structured answers with clear source attribution.
Built using **LangChain**, **FAISS**, **Groq**, **Search1API**, and **Streamlit**.

---

## Features

- **Ask Questions About Any Webpage** → paste a URL and query its content.
- **Dual Knowledge Sources**:
  - `[RAG]` → extracts and analyzes information directly from the given webpage.  
  - `[Search]` → enriches answers with real-time search results.
- **Clear, Structured Answers** → concise bullet-point responses.
- **Source Attribution** → always shows where the information came from.
- **Streamlit Interface** → clean, interactive, and easy to use.

---

## Tech Stack

- **LangChain** – Orchestrates document loading, retrieval, and prompt formatting.
- **FAISS** – Stores vector embeddings for fast similarity search.
- **Groq LLM** – Generates responses based on combined context.
- **Search1API** – Provides live web search results.
- **Streamlit** – Frontend interface for user interaction.
- **HuggingFace Embeddings** – Converts text into vector embeddings.

---

## Setup

1. **Clone the repository**
```bash
git clone <your_repo_url>
cd <repo_folder>
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create .env file in project root:**
```bash
GROQ_API_KEY=<your_groq_api_key>
SEARCH1API_KEY=<your_search1api_key>
```

5. **Run the app**
```bash
streamlit run app.py
```

## Usage

1. Enter a webpage URL into the input box.

2. Ask a question related to the page content.

3. View the answer with sources:
   - `[RAG]` → content retrieved from the page
   - `[Search]` → external search results


## File structure

File Structure
├── app.py              # Streamlit frontend
├── rag.py              # RAG chain setup and Search1API integration
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md