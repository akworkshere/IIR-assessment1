# Project Report Analyzer

## Video Demo

Link: [Public Google Drive Link](https://drive.google.com/file/d/1Ulmn8nN1ai_axk3DcbOH1M9uPkvvayXp/view?usp=sharing)

## Prerequisites

Before running this project, ensure you have:

1. **Python 3.10 or higher** installed
2. **Ollama** installed and running with `phi3:mini` model

---

## Step-by-Step Setup Instructions

### Step 1: Install Ollama (if not already installed)

1. Download & install Ollama from: https://ollama.ai/download
2. Open a terminal and pull the required model:
   ```bash
   ollama pull phi3:mini
   ```
3. Verify Ollama is running:
   ```bash
   ollama list
   ```
---

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

---

### Step 3: Activate Virtual Environment

**On Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**On Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Verify Ollama is Running

Before starting the app, make sure Ollama is running:

```bash
ollama list
```

If it shows `phi3:mini`, you're good to go

---

### Step 6: Run the Application

```bash
streamlit run app.py
```

---

### Step 7: Access the Application

- Your browser should automatically open to: **http://localhost:8501**
- If not, manually open your browser and go to that URL

---

## Using the Application

### Upload Documents
1. Click the **"Upload Documents"** tab
2. Click **"Browse files"** or drag-and-drop PDF files
3. Click **"Process Documents"** button
4. Wait for processing to complete

### Ask Questions
1. Click the **"Ask Questions"** tab
2. Type your question in the text area (or click a sample question)
3. Click **"Ask"** button

---

## Tech Stack

| Category | Technology | Why This Choice |
|----------|------------|-----------------|
| **Web Framework** | Streamlit | Rapid prototyping for ML apps; built-in session state management; no frontend code needed |
| **PDF Processing** | pdfplumber | Superior table extraction; handles complex layouts; better accuracy than PyPDF2 for text |
| **Embeddings** | sentence-transformers (all-mpnet-base-v2) | State-of-the-art semantic search; 768-dim vectors; good balance of speed and accuracy |
| **Vector Database** | FAISS (HNSW index) | Facebook's proven similarity search; handles millions of vectors; fast approximate nearest neighbor |
| **LLM** | Phi-3 Mini via Ollama | Runs locally (no API costs); 3.8B params yet competitive quality; fast inference on CPU |
| **Text Chunking** | LangChain RecursiveCharacterTextSplitter | Smart splitting at sentence/paragraph boundaries; configurable overlap for context preservation |
| **LLM Runtime** | Ollama | Easy local model management; cross-platform; simple REST API |

### Architecture Flow

<img src="Generated%20Board.png" alt="Architecture Diagram" width="600">

---
