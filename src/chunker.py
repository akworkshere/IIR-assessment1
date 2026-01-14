from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_document(self, processed_doc: Dict) -> List[Dict]:
        if not processed_doc.get("success"):
            logger.error("Cannot chunk unsuccessful document processing result")
            return []

        chunks = []
        document_name = processed_doc["document_name"]
        file_path = processed_doc.get("file_path", "")

        for page in processed_doc["pages"]:
            page_num = page["page_num"]
            page_text = page["text"]

            if page.get("has_tables"):
                for table in page["tables"]:
                    page_text += "\n\n" + table["text"]

            if not page_text.strip():
                continue

            page_chunks = self.text_splitter.split_text(page_text)

            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "document_name": document_name,
                        "file_path": file_path,
                        "page_num": page_num,
                        "chunk_id": f"{document_name}_p{page_num}_c{chunk_idx}",
                        "chunk_index": chunk_idx,
                        "total_pages": processed_doc["total_pages"],
                        "has_tables": page.get("has_tables", False),
                    }
                }
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from {document_name} ({processed_doc['total_pages']} pages)")

        return chunks

    def chunk_multiple_documents(self, processed_docs: List[Dict]) -> List[Dict]:
        all_chunks = []

        for doc in processed_docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(processed_docs)} documents")

        return all_chunks

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {"total_chunks": 0}

        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
        documents = set(chunk["metadata"]["document_name"] for chunk in chunks)

        return {
            "total_chunks": len(chunks),
            "total_documents": len(documents),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths),
            "documents": list(documents),
        }


def chunk_processed_document(processed_doc: Dict, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(processed_doc)
