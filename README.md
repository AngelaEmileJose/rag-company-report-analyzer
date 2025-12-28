# RAG Company Report Analyzer

A powerful Retrieval-Augmented Generation (RAG) system designed to analyze company financial reports, executive summaries, and business documents. Built with **FastAPI**, **LangChain**, and **Groq**, this tool extracts insights, generates strategic questions, and creates executive summaries from uploaded documents (PDF/Text) or URLs.

## 🚀 Features

-   **Intelligent Document Processing**: Automatically extracts text from PDFs and websites.
-   **Context-Aware Question Generation**: Generates specific, high-quality questions based on the actual content of the document (e.g., specific revenue figures, risks, operational milestones).
-   **Executive Summary Generation**: Produces comprehensive summaries covering Financial Performance, Operational Updates, Future Outlook, Risks, and Strategic Initiatives.
-   **Interactive Q&A**: Chat with your documents to get precise answers with source references.
-   **High Performance**: Uses `FAISS` for fast vector retrieval and caching for optimized processing.
-   **PDF Export**: Export generated Q&A and summaries to professional PDF reports.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI
-   **AI/ML**: LangChain, SentenceTransformers, Groq API (Llama 3 70B)
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Frontend**: HTML/CSS/JavaScript (served via FastAPI StaticFiles)
-   **Utilities**: PyMuPDF (PDF processing), ReportLab (PDF generation)

## 📂 Project Structure

```
rag-company-report-analyzer/
├── api_server.py           # Main FastAPI entry point
├── rag_system/
│   ├── core.py             # Main RAG logic (Ingestion, QA, Generation)
│   ├── config.py           # Configuration management (Pydantic models)
│   ├── document_processor.py # PDF and URL text extraction
│   ├── models.py           # Data models
│   └── utils.py            # Helper functions (logging, etc.)
├── static/                 # Frontend assets (HTML, CSS, JS)
├── data/                   # Directory for storing uploaded/processed data
├── cache/                  # Cache for processed documents and vector stores
├── logs/                   # Application logs
└── requirements.txt        # Python dependencies
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd rag-company-report-analyzer
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=gsk_your_groq_api_key_here
    # Optional overlays
    MODEL_NAME=llama-3.3-70b-versatile
    LOG_LEVEL=INFO
    ```

## 🚀 Usage

1.  **Start the Server**:
    ```bash
    python api_server.py
    ```
    The server will start on `http://0.0.0.0:8000`.

2.  **Access the Web Interface**:
    Open your browser and navigate to `http://localhost:8000`.

3.  **Analyze a Document**:
    -   **Upload PDF**: Use the upload button to select a financial report.
    -   **Enter URL**: Paste a link to a public webpage (note: pages requiring login cannot be processed).
    -   Click "Process" to ingest the document.

4.  **Generate Insights**:
    -   **Generate Questions**: Click "Generate Questions" to get AI-suggested questions based on the document.
    -   **Chat**: Type specific questions in the chat box.
    -   **Summary**: Click "Generate Summary" for a full executive overview.

## 🔧 Configuration

The system is highly configurable via `rag_system/config.py` or `.env` variables:

-   `CHUNK_SIZE`: Size of text chunks for splitting (default: 1000).
-   `CHUNK_OVERLAP`: Overlap between chunks (default: 200).
-   `RETRIEVAL_K`: Number of document chunks to retrieve for answers (default: 8).
-   `ENABLE_CACHE`: Enable/disable caching of processed texts and vector stores.

## 🤝 Contributing

Contributions are welcome! Please ensure you test any changes to the `generate_questions` logic in `rag_system/core.py` to maintain high quality.