import os
import streamlit as st
from src.processor import DocumentProcessor
from src.database import VectorDatabase
from src.engine import RAGEngine

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# UI dla klucza API
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.selectbox("Model LLM", MODEL_OPTIONS)
retrieval_mode = st.sidebar.selectbox("Retrieval", ["similarity", "mmr"])
top_k = st.sidebar.slider("Top-k", 1, 10, 3)

if api_key:
    # Inicjalizacja komponentów
    db = VectorDatabase(api_key)
    processor = DocumentProcessor()
    engine = RAGEngine(api_key, model_name=model_name)

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = st.file_uploader("Dodaj PDF do bazy", type="pdf")

    if uploaded_file:
        # Zapis i procesowanie
        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/{uploaded_file.name}"

        if uploaded_file.name in st.session_state.processed_files:
            st.info("Ten plik jest już w bazie")
        else:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            chunks, stats = processor.process_pdf(file_path)
            db.create_or_update(chunks)
            st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"Dodano {stats['filename']} ({stats['pages']} stron) chunki: {stats['num_chunks']}")

    # Chat
    retriever = db.load_retriever(mode=retrieval_mode, k=top_k)
    if retriever:

        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

        if prompt := st.chat_input("Zadaj pytanie"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            chain = engine.get_chain(retriever)
            response = chain.invoke({"input": prompt})
            answer = response.get("answer", "")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

    else:
        st.warning("Brak indeksu. Dodaj PDF aby uruchomić czat")
        st.stop()