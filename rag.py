import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEARCH1API_KEY = os.getenv("SEARCH1API_KEY")

INDEX_DIR = "faiss_index"  # folder to persist FAISS index


def search_search1api(query: str, max_results: int = 3) -> list:
    """Search external web using Search1API (DuckDuckGo/Google/Bing)."""
    url = "https://api.search1api.com/search"
    headers = {
        "Authorization": f"Bearer {SEARCH1API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "query": query,
        "search_service": "google", # "duckduckgo", "bing"
        "max_results": 5,
        "crawl_results": 0,
        "image": False,
        "language": "en",
        "include_sites": [],
        "exclude_sites": [],
        "language": "en",
        "time_range": "year"
    }

    try:
        response = requests.request("POST", url, json=json_data, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append({"title": title, "snippet": snippet, "link": link})

        return results
    except Exception as e:
        print(f"Search1API error: {e}")
        return []


def setup_rag_chain(url: str, persist: bool = True):
    """Load webpage, embed, store in FAISS (with persistence)."""
    try:
        # if already saved, load from disk
        if persist and os.path.exists(INDEX_DIR):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True).as_retriever(
                search_kwargs={"k": 5}
            )

        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)

        if persist:
            vector_store.save_local(INDEX_DIR)

        return vector_store.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"Error in setup_rag_chain: {e}")
        return None
