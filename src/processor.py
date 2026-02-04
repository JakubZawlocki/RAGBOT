import os
import statistics
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
        file_name = os.path.basename(file_path)


        for i, pdf in enumerate(pages):
            pdf.metadata["source"] = file_name
            pdf.metadata["page"] = int(pdf.metadata.get("page", i)) + 1


        # Statystyki do Punktu 4
        chunks = self.splitter.split_documents(pages)
        chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()] # likwidacja chunków wypełnionych pustymi znakami
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        stats = {
            "filename": file_name,
            "pages": len(pages),
            "total_chars": sum(len(p.page_content) for p in pages),
            "num_chunks": len(chunks),
            "chunk_min": min(chunk_lengths) if chunk_lengths else 0,
            "chunk_avg": int(sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0,
            "chunk_median": int(statistics.median(chunk_lengths)) if chunk_lengths else 0,
            "chunk_max": max(chunk_lengths) if chunk_lengths else 0
        }
        
        return chunks, stats