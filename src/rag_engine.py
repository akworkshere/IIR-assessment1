from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time

from .document_processor import DocumentProcessor
from .chunker import DocumentChunker
from .embeddings import EmbeddingGenerator
from .vector_store import FAISSVectorStore
from .llm import LLMClient

logger = logging.getLogger(__name__)


class RAGEngine:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model: str = "phi3:mini",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 512,
        cache_dir: Path = None
    ):
        logger.info("Initializing RAG Engine...")

        self.document_processor = DocumentProcessor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            cache_dir=cache_dir
        )
        self.llm = LLMClient(
            model_name=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )

        embedding_dim = self.embedding_generator.get_embedding_dim()
        self.vector_store = FAISSVectorStore(embedding_dim=embedding_dim)

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.processed_documents: List[Dict] = []
        self.processed_document_names: set = set()

        self.persistent_session_dir = Path("data/vector_stores/persistent_session")
        self.persistent_session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RAG Engine initialized successfully")

    def add_documents(self, pdf_paths: List[Path], show_progress: bool = True) -> Dict:
        logger.info(f"Adding {len(pdf_paths)} documents")

        results = {
            "success": True,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "total_chunks": 0
        }

        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                if pdf_path.name in self.processed_document_names:
                    logger.info(f"Skipping already processed document: {pdf_path.name}")
                    results["skipped"] += 1
                    continue

                logger.info(f"Processing: {pdf_path.name}")
                processed_doc = self.document_processor.process_pdf(pdf_path)

                if not processed_doc["success"]:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": pdf_path.name,
                        "error": processed_doc["error"]
                    })
                    continue

                chunks = self.chunker.chunk_document(processed_doc)

                if not chunks:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": pdf_path.name,
                        "error": "No chunks created"
                    })
                    continue

                chunks_with_embeddings = self.embedding_generator.embed_chunks(
                    chunks,
                    show_progress=show_progress
                )

                all_chunks.extend(chunks_with_embeddings)
                self.processed_documents.append(processed_doc)
                self.processed_document_names.add(pdf_path.name)
                results["processed"] += 1

                logger.info(f"Successfully processed {pdf_path.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({
                    "file": pdf_path.name,
                    "error": str(e)
                })

        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            results["total_chunks"] = len(all_chunks)

        if results["failed"] > 0:
            results["success"] = False

        logger.info(f"Document addition complete: {results['processed']} successful, {results['failed']} failed")

        if results["processed"] > 0:
            self._save_persistent_session()

        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        document_filter: Optional[str] = None
    ) -> Dict:
        logger.info(f"Processing query: {question[:100]}...")
        start_time = time.time()

        if len(self.vector_store.chunks) == 0:
            return {
                "success": False,
                "error": "no_documents",
                "message": "Please upload at least one document before asking questions."
            }

        try:
            if top_k is None:
                top_k = self.top_k

            embedding_start = time.time()
            query_embedding = self.embedding_generator.embed_text(question)
            embedding_time = time.time() - embedding_start

            retrieval_start = time.time()
            if document_filter:
                results = self.vector_store.search_by_metadata(
                    query_embedding=query_embedding,
                    metadata_filter={"document_name": document_filter},
                    top_k=top_k
                )
            else:
                results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    similarity_threshold=self.similarity_threshold
                )
            retrieval_time = time.time() - retrieval_start

            if not results:
                total_time = time.time() - start_time
                return {
                    "success": True,
                    "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "relevance": "low",
                    "timing": {
                        "total": round(total_time, 2),
                        "embedding": round(embedding_time, 3),
                        "retrieval": round(retrieval_time, 3),
                        "generation": 0
                    }
                }

            context, sources = self._build_context(results)

            generation_start = time.time()
            answer = self._generate_answer(question, context)
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time

            estimated_prompt_tokens = (len(question) + len(context)) // 4
            estimated_completion_tokens = len(answer) // 4

            relevance_level, relevance_reason = self._assess_relevance(results)

            response = {
                "success": True,
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(results),
                "relevance": relevance_level,
                "relevance_reason": relevance_reason,
                "question": question,
                "timing": {
                    "total": round(total_time, 2),
                    "embedding": round(embedding_time, 3),
                    "retrieval": round(retrieval_time, 3),
                    "generation": round(generation_time, 2)
                },
                "tokens": {
                    "prompt": estimated_prompt_tokens,
                    "completion": estimated_completion_tokens,
                    "total": estimated_prompt_tokens + estimated_completion_tokens
                }
            }

            logger.info(f"Query processed successfully. Retrieved {len(results)} chunks in {total_time:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": "processing_error",
                "message": f"Error processing query: {str(e)}"
            }

    def _build_context(self, results: List[Tuple[Dict, float]]) -> Tuple[str, List[Dict]]:
        context_parts = []
        sources = []

        for idx, (chunk, similarity) in enumerate(results, start=1):
            metadata = chunk["metadata"]

            chunk_text = f"[Source: {metadata['document_name']}, Page {metadata['page_num']}]\n{chunk['text']}"
            context_parts.append(chunk_text)

            sources.append({
                "document": metadata["document_name"],
                "page": metadata["page_num"],
                "chunk_id": metadata["chunk_id"],
                "similarity": round(similarity, 3),
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            })

        context = "\n\n---\n\n".join(context_parts)

        return context, sources

    def _generate_answer(self, question: str, context: str) -> str:
        system_prompt = """You are a helpful AI assistant analyzing project reports.
Your task is to answer questions accurately based ONLY on the provided context.

Key Guidelines:
1. Always cite your sources using [Source: DOCUMENT_NAME, Page X] format
2. If information is not in the context, clearly state "I don't have this information in the provided documents"
3. For multi-document answers, organize by project/document
4. Include relevant direct quotes when they add value
5. Be concise but comprehensive
6. If there are conflicting information across documents, present both and note the conflict"""

        prompt = f"""Context from project reports:
---
{context}
---

Question: {question}

Instructions:
- Answer based solely on the context above
- Cite each fact with [Source: DOCUMENT_NAME, Page X]
- If the context doesn't contain sufficient information, say so
- Provide specific details (numbers, dates, names) when available

Answer:"""

        answer = self.llm.generate(prompt=prompt, system_prompt=system_prompt)

        return answer

    def _assess_relevance(self, results: List[Tuple[Dict, float]]) -> Tuple[str, str]:
        if not results:
            return "none", "No relevant chunks found"

        avg_similarity = sum(score for _, score in results) / len(results)
        max_similarity = max(score for _, score in results)
        min_similarity = min(score for _, score in results)

        if avg_similarity > 0.6:
            reason = f"Strong semantic match (avg: {avg_similarity:.1%}, best: {max_similarity:.1%})"
            return "high", reason
        elif avg_similarity > 0.4:
            reason = f"Moderate semantic match (avg: {avg_similarity:.1%}, range: {min_similarity:.1%}-{max_similarity:.1%})"
            return "medium", reason
        else:
            reason = f"Weak semantic match (avg: {avg_similarity:.1%}). The answer may be less accurate or based on tangentially related content."
            return "low", reason

    def clear_documents(self) -> None:
        self.vector_store.clear()
        self.processed_documents = []
        self.processed_document_names.clear()

        self._delete_persistent_session()

        logger.info("Cleared all documents and persistent session")

    def get_stats(self) -> Dict:
        vector_stats = self.vector_store.get_stats()

        return {
            "total_documents": len(self.processed_documents),
            "total_chunks": vector_stats["total_chunks"],
            "documents": [doc["document_name"] for doc in self.processed_documents],
            "embedding_dim": self.embedding_generator.get_embedding_dim(),
            "vector_store": vector_stats
        }

    def save_vector_store(self, save_dir: Path) -> None:
        self.vector_store.save(save_dir)
        logger.info(f"Saved vector store to {save_dir}")

    def load_vector_store(self, load_dir: Path) -> None:
        self.vector_store = FAISSVectorStore.load(load_dir)
        logger.info(f"Loaded vector store from {load_dir}")

    def _save_persistent_session(self) -> None:
        try:
            self.vector_store.save(self.persistent_session_dir)

            metadata_file = self.persistent_session_dir / "session_metadata.pkl"
            import pickle
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    "processed_documents": self.processed_documents,
                    "processed_document_names": list(self.processed_document_names)
                }, f)

            logger.info("Persistent session saved successfully")
        except Exception as e:
            logger.error(f"Failed to save persistent session: {str(e)}")

    def load_persistent_session(self) -> bool:
        try:
            index_file = self.persistent_session_dir / "faiss.index"
            metadata_file = self.persistent_session_dir / "session_metadata.pkl"

            if not (index_file.exists() and metadata_file.exists()):
                logger.info("No persistent session found")
                return False

            self.vector_store = FAISSVectorStore.load(self.persistent_session_dir)

            import pickle
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.processed_documents = data["processed_documents"]
                self.processed_document_names = set(data["processed_document_names"])

            logger.info(f"Persistent session loaded: {len(self.processed_documents)} documents, {len(self.vector_store.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to load persistent session: {str(e)}")
            return False

    def _delete_persistent_session(self) -> None:
        try:
            import shutil
            if self.persistent_session_dir.exists():
                shutil.rmtree(self.persistent_session_dir)
                self.persistent_session_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Persistent session deleted")
        except Exception as e:
            logger.error(f"Failed to delete persistent session: {str(e)}")

    def has_persistent_session(self) -> bool:
        index_file = self.persistent_session_dir / "faiss.index"
        metadata_file = self.persistent_session_dir / "session_metadata.pkl"
        return index_file.exists() and metadata_file.exists()


def create_rag_engine(**kwargs) -> RAGEngine:
    return RAGEngine(**kwargs)
