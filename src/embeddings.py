from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: Path = None,
        device: str = None
    ):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading embedding model: {model_name} on {device}")

        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir) if cache_dir else None,
                device=device
            )

            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"Successfully loaded model. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.embedding_dim)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim))

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return np.zeros((len(texts), self.embedding_dim))

    def embed_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[dict]:
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        logger.info(f"Successfully generated {len(chunks)} embeddings")

        return chunks

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(max(0.0, similarity))


_global_embedding_generator = None


def get_embedding_generator(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    cache_dir: Path = None,
    device: str = None
) -> EmbeddingGenerator:
    global _global_embedding_generator

    if _global_embedding_generator is None:
        _global_embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )

    return _global_embedding_generator


def embed_text(text: str) -> np.ndarray:
    generator = get_embedding_generator()
    return generator.embed_text(text)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    generator = get_embedding_generator()
    return generator.embed_batch(texts, batch_size=batch_size)
