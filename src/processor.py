import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Statystyki do Punktu 4
        stats = {
            "filename": os.path.basename(file_path),
            "pages": len(pages),
            "total_chars": sum(len(p.page_content) for p in pages)
        }
        
        chunks = self.splitter.split_documents(pages)
        return chunks, stats