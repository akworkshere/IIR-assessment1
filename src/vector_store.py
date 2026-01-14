import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "HNSW",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.chunks: List[Dict] = []
        self.doc_count = 0

        logger.info(f"Initialized FAISS {index_type} index with dimension {embedding_dim}")

    def add_chunks(self, chunks: List[Dict]) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return

        if not all("embedding" in chunk for chunk in chunks):
            raise ValueError("All chunks must have 'embedding' field")

        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype('float32')

        self.index.add(embeddings)

        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[Dict, float]]:
        if len(self.chunks) == 0:
            logger.warning("No chunks in vector store")
            return []

        query_vector = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            similarity = self._distance_to_similarity(distance)

            if similarity < similarity_threshold:
                continue

            chunk = self.chunks[idx].copy()
            results.append((chunk, similarity))

        logger.debug(f"Search returned {len(results)} results (top_k={top_k})")

        return results

    def _distance_to_similarity(self, distance: float) -> float:
        return 1.0 / (1.0 + distance)

    def search_by_metadata(
        self,
        query_embedding: np.ndarray,
        metadata_filter: Dict,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        results = self.search(query_embedding, top_k=top_k * 3)

        filtered_results = []
        for chunk, score in results:
            matches = all(
                chunk["metadata"].get(key) == value
                for key, value in metadata_filter.items()
            )
            if matches:
                filtered_results.append((chunk, score))

            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        metadata_path = save_dir / "chunks.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)

        logger.info(f"Saved vector store to {save_dir}")

    @classmethod
    def load(cls, load_dir: Path) -> 'FAISSVectorStore':
        load_dir = Path(load_dir)

        index_path = load_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")

        metadata_path = load_dir / "chunks.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"]
        )

        instance.index = faiss.read_index(str(index_path))
        instance.chunks = data["chunks"]

        logger.info(f"Loaded vector store from {load_dir} with {len(instance.chunks)} chunks")

        return instance

    def clear(self) -> None:
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)

        self.chunks = []
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "index_size": self.index.ntotal,
            "documents": list(set(
                chunk["metadata"]["document_name"] for chunk in self.chunks
            ))
        }


def create_vector_store(embedding_dim: int, index_type: str = "HNSW") -> FAISSVectorStore:
    return FAISSVectorStore(embedding_dim=embedding_dim, index_type=index_type)
