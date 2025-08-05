import streamlit as st
import os
from rag import search_search1api, setup_rag_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Intellecta", layout="centered")
st.title("Intellecta")
st.caption("Enter a URL, then ask a question")
st.image("my_img.jpg")

# Session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""

url = st.text_input("Enter a webpage URL", key="url")

if url:
    if url != st.session_state.processed_url:
        with st.spinner("Processing URL..."):
            st.session_state.retriever = setup_rag_chain(url)
            st.session_state.processed_url = url
            if st.session_state.retriever:
                st.success("URL processed successfully!")
            else:
                st.error("Failed to process this URL.")

    if st.session_state.retriever:
        st.markdown("---")
        question = st.text_input("Ask a question about this page:")
        if question:
            with st.spinner("Finding answer..."):
                try:
                    # Retrieve RAG docs
                    retrieved_docs = st.session_state.retriever.invoke(question)
                    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])

                    # Web search
                    search_results = search_search1api(question)

                    # Final context
                    final_text = f"RAG DATA:\n{rag_context}\n\nWEB SEARCH:\n{search_results}"

                    # LLM
                    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
                    prompt = ChatPromptTemplate.from_template(
                        """
                        Answer the question using the context below.
                        - Use bullet points only
                        - Indicate source: [RAG] or [Search]
                        - If missing, say you couldn't find it

                        Context:
                        {context}

                        Question: {input}

                        Answer:
                        """
                    )

                    formatted_prompt = prompt.format(context=final_text, input=question)
                    response = llm.invoke(formatted_prompt)

                    st.markdown("**Answer:**")
                    st.success(response.content)

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("Enter a URL above to get started.")

st.markdown("---")
st.caption("Powered by LangChain, Groq, FAISS, Search1API")
