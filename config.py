import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
CACHE_DIR = DATA_DIR / "cache"

for dir_path in [DATA_DIR, UPLOAD_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "phi3:mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.3
ENABLE_RERANKING = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 3

FAISS_INDEX_TYPE = "HNSW"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50

MAX_UPLOAD_SIZE_MB = 50
SUPPORTED_FORMATS = [".pdf"]
MIN_TEXT_LENGTH = 50

STREAMLIT_PAGE_TITLE = "Project Report Analyzer"
STREAMLIT_PAGE_ICON = "📄"
STREAMLIT_LAYOUT = "wide"
MAX_QUERY_HISTORY = 50

SYSTEM_PROMPT = """You are a helpful AI assistant analyzing project reports.
Your task is to answer questions accurately based ONLY on the provided context.

Key Guidelines:
1. Always cite your sources using [Source: DOCUMENT_NAME, Page X] format
2. If information is not in the context, clearly state "I don't have this information in the provided documents"
3. For multi-document answers, organize by project/document
4. Include relevant direct quotes when they add value
5. Be concise but comprehensive
6. If there are conflicting information across documents, present both and note the conflict
"""

QA_PROMPT_TEMPLATE = """Context from project reports:
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

ERROR_MESSAGES = {
    "invalid_pdf": "Unable to process this PDF. Please ensure it's a valid, text-based PDF (not a scanned image).",
    "empty_pdf": "This PDF appears to be empty or contains no extractable text.",
    "file_too_large": f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.",
    "no_documents": "Please upload at least one document before asking questions.",
    "irrelevant_question": "This question doesn't appear to relate to the uploaded documents. Please ask about project details, timelines, budgets, or other report contents.",
    "insufficient_info": "I couldn't find sufficient information in the documents to answer this question.",
    "ollama_connection": "Cannot connect to Ollama. Please ensure Ollama is running and phi3:mini model is installed.",
    "embedding_model_load": "Failed to load embedding model. Please check your internet connection for first-time download.",
}

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
