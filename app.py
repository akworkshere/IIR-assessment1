import streamlit as st
from pathlib import Path
import sys
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_engine import RAGEngine
from src.llm import check_ollama_status
import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="QNA System",
    page_icon="📄",
    layout=config.STREAMLIT_LAYOUT
)


def initialize_session_state():
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""


def check_system_requirements():
    status = check_ollama_status()

    if status["status"] == "offline":
        st.error("Ollama is not running")
        st.markdown("""
        **Please start Ollama:**
        """)
        return False

    if not any("phi3" in model for model in status["models"]):
        st.warning("Phi-3 Mini model not found!")
        st.markdown("""
        **Install the model:**
        ```bash
        ollama pull phi3:mini
        ```
        """)
        return False

    return True


def initialize_rag_engine():
    try:
        with st.spinner("Initializing RAG Engine (first time may download models)..."):
            rag_engine = RAGEngine(
                embedding_model=config.EMBEDDING_MODEL,
                llm_model=config.LLM_MODEL,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                top_k=config.TOP_K_CHUNKS,
                similarity_threshold=config.SIMILARITY_THRESHOLD,
                llm_temperature=config.LLM_TEMPERATURE,
                llm_max_tokens=config.LLM_MAX_TOKENS,
                cache_dir=config.CACHE_DIR
            )

        st.session_state.rag_engine = rag_engine
        st.session_state.system_ready = True

        session_loaded = rag_engine.load_persistent_session()
        if session_loaded:
            st.session_state.documents_loaded = True
            st.success("System initialized successfully! Previous session restored.")
        else:
            st.success("System initialized successfully!")

        return True

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return False


def render_sidebar():
    st.sidebar.title("Document Manager")

    if st.session_state.system_ready and st.session_state.rag_engine:
        stats = st.session_state.rag_engine.get_stats()

        st.sidebar.metric("Documents Loaded", stats["total_documents"])
        st.sidebar.metric("Total Chunks", stats["total_chunks"])

        if st.session_state.rag_engine.has_persistent_session():
            st.sidebar.success("✓ Session Persisted")
        else:
            st.sidebar.info("No saved session")

        if stats["documents"]:
            st.sidebar.subheader("Loaded Documents")
            for doc in stats["documents"]:
                st.sidebar.text(doc)

        if st.sidebar.button("Clear All Documents", type="secondary"):
            st.session_state.rag_engine.clear_documents()
            st.session_state.documents_loaded = False
            st.session_state.query_history = []
            st.rerun()

    else:
        st.sidebar.warning("System not initialized")

    st.sidebar.markdown("---")

    with st.sidebar.expander("Settings"):
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=config.TOP_K_CHUNKS,
            help="More chunks = more context but slower"
        )

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.SIMILARITY_THRESHOLD,
            step=0.1,
            help="Lower = more permissive, higher = more strict"
        )

        if st.session_state.rag_engine:
            st.session_state.rag_engine.top_k = top_k
            st.session_state.rag_engine.similarity_threshold = similarity_threshold


def render_document_upload():
    st.header("Upload Project Reports")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF project reports"
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            pdf_paths = []

            with st.spinner("Saving uploaded files..."):
                for uploaded_file in uploaded_files:
                    file_path = config.UPLOAD_DIR / uploaded_file.name

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pdf_paths.append(file_path)

            with st.spinner("Processing PDFs (extracting text, generating embeddings)..."):
                results = st.session_state.rag_engine.add_documents(pdf_paths)

            if results["success"]:
                st.success(f"Successfully processed {results['processed']} documents!")
                if results["skipped"] > 0:
                    st.info(f"Skipped {results['skipped']} already processed document(s)")
                st.info(f"Created {results['total_chunks']} new searchable chunks")
                st.session_state.documents_loaded = True

            else:
                st.warning(f"Processed {results['processed']} documents, {results['failed']} failed")
                if results["skipped"] > 0:
                    st.info(f"Skipped {results['skipped']} already processed document(s)")

                if results["errors"]:
                    with st.expander("View Errors"):
                        for error in results["errors"]:
                            st.error(f"**{error['file']}**: {error['error']}")

            st.rerun()


def render_query_interface():
    st.header("Ask Questions")

    if not st.session_state.documents_loaded:
        st.info("Please upload documents first")
        return

    st.subheader("Sample Questions")

    sample_questions = {
        "Single Document": [
            "What is the location of the Freeport refinery project?",
            "What was the total investment value for the Xianyang plant?",
            "Who was the project manager for the Racine power station?"
        ],
        "Multi-Document": [
            "Compare the budgets of all three projects",
            "Which projects were cancelled and why?",
            "What are the different locations of these projects?",
            "Compare the timelines from initial release to last update"
        ]
    }

    for category, questions in sample_questions.items():
        st.markdown(f"**{category}:**")
        for idx, question in enumerate(questions):
            if st.button(question, key=f"q_{category}_{idx}", use_container_width=False):
                st.session_state.selected_question = question
                st.rerun()

    st.markdown("---")

    question = st.text_area(
        "Enter your question:",
        value=st.session_state.selected_question,
        placeholder="e.g., Compare the budgets of these projects",
        height=100,
        key="question_input"
    )

    if st.session_state.selected_question:
        st.session_state.selected_question = ""

    col1, col2 = st.columns([1, 4])

    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    with col2:
        clear_history = st.button("Clear History", use_container_width=True)
        if clear_history:
            st.session_state.query_history = []
            st.rerun()

    if ask_button and question.strip():
        with st.spinner("Thinking..."):
            response = st.session_state.rag_engine.query(question)

            query_record = {
                "question": question,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "relevance": response.get("relevance", "unknown")
            }
            st.session_state.query_history.append(query_record)

        render_response(response)

    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("Recent Queries")

        for query in reversed(st.session_state.query_history[-3:]):
            with st.expander(f"Q: {query['question'][:100]}"):
                st.text(f"Time: {query['timestamp']}")
                render_response(query['response'], show_sources_in_expander=False)


def render_response(response: dict, show_sources_in_expander: bool = True):
    if not response["success"]:
        st.error(f"Error: {response.get('message', 'Unknown error')}")
        return

    st.markdown("### Answer")
    st.markdown(response["answer"])

    if "timing" in response or "tokens" in response:
        col1, col2, col3 = st.columns(3)

        if "timing" in response:
            with col1:
                st.metric("Total Time", f"{response['timing']['total']}s")
            with col2:
                st.metric("Retrieval", f"{response['timing']['retrieval']}s")
            with col3:
                st.metric("Generation", f"{response['timing']['generation']}s")

        if "tokens" in response:
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Prompt Tokens", f"{response['tokens']['prompt']:,}")
            with col5:
                st.metric("Completion Tokens", f"{response['tokens']['completion']:,}")

    relevance = response.get("relevance", "unknown")
    relevance_reason = response.get("relevance_reason", "")

    if relevance == "high":
        st.success("High confidence answer")
        if relevance_reason:
            st.caption(f"Details: {relevance_reason}")
    elif relevance == "medium":
        st.info("Medium confidence answer")
        if relevance_reason:
            st.caption(f"Details: {relevance_reason}")
    elif relevance == "low":
        st.warning("Low confidence - answer may be less accurate")
        if relevance_reason:
            st.caption(f"Reason: {relevance_reason}")

    if response.get("sources"):
        st.markdown("### Sources")

        sources_by_doc = {}
        for source in response["sources"]:
            doc_name = source["document"]
            if doc_name not in sources_by_doc:
                sources_by_doc[doc_name] = []
            sources_by_doc[doc_name].append(source)

        if show_sources_in_expander:
            for doc_name, sources in sources_by_doc.items():
                with st.expander(f"{doc_name}"):
                    for source in sources:
                        st.markdown(f"""
                        **Page {source['page']}** (Similarity: {source['similarity']:.2%})
                        > {source['text_preview']}
                        """)
        else:
            for doc_name, sources in sources_by_doc.items():
                st.markdown(f"**{doc_name}:**")
                for source in sources:
                    st.markdown(f"""
                    - **Page {source['page']}** (Similarity: {source['similarity']:.2%})
                    """)
                    with st.container():
                        st.caption(f"> {source['text_preview'][:150]}...")

        st.text(f"Retrieved {len(response['sources'])} relevant chunks")


def main():
    initialize_session_state()

    st.title("QNA System")

    if not check_system_requirements():
        st.stop()

    if not st.session_state.system_ready:
        if not initialize_rag_engine():
            st.stop()

    render_sidebar()

    tab1, tab2 = st.tabs(["Upload Documents", "Ask Questions"])

    with tab1:
        render_document_upload()

    with tab2:
        render_query_interface()

    st.markdown("---")


if __name__ == "__main__":
    main()
