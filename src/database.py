from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

class VectorDatabase:
    def __init__(self, api_key, storage_path="faiss_index"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.storage_path = storage_path

    def create_or_update(self, chunks):
        if os.path.exists(self.storage_path):
            vector_store = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        vector_store.save_local(self.storage_path)
        return vector_store

    def load_retriever(self, mode="similarity", k=3):
        if not os.path.exists(self.storage_path):
            return None
        
        vs = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
        
        if mode == "mmr":
            return vs.as_retriever(search_type="mmr", search_kwargs={"k":k, "fetch_k": max(20, k*5)})
        
        return vs.as_retriever(search_kwargs={"k":k})