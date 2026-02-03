import streamlit as st
from src.processor import DocumentProcessor
from src.database import VectorDatabase
from src.engine import RAGEngine

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# UI dla klucza API
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if api_key:
    # Inicjalizacja komponentów
    db = VectorDatabase(api_key)
    processor = DocumentProcessor()
    engine = RAGEngine(api_key)

    uploaded_file = st.file_uploader("Dodaj PDF do bazy", type="pdf")
    
    if uploaded_file:
        # Zapis i procesowanie
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        chunks, stats = processor.process_pdf(f"data/{uploaded_file.name}")
        db.create_or_update(chunks)
        st.success(f"Dodano {stats['filename']} ({stats['pages']} stron)")

    # Chat
    retriever = db.load_retriever()
    if retriever:
        if prompt := st.chat_input("Zadaj pytanie"):
            st.chat_message("user").write(prompt)
            
            chain = engine.get_chain(retriever)
            response = chain.invoke({"input": prompt})
            
            with st.chat_message("assistant"):
                st.write(response["answer"])