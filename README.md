Project Report: RAG Company Report Analyzer
1. Overview
The RAG Company Report Analyzer is a Python-based application designed to process corporate reports (in PDF format) and enable advanced question-answering over the extracted content. It uses:

LangChain for orchestration of LLM interactions

FAISS for efficient vector-based document retrieval

Groq LLMs for fast, large-scale inference

Streamlit for a user-friendly web interface

The system supports:

CLI execution (rag_question_cli.py)

Web UI (web_interface.py)

Core backend logic in RAG.py

2. Features
Load PDF documents from:

Direct URL

Local file upload

Automatically split documents into semantic chunks

Embed document content using sentence-transformers

Store embeddings in a FAISS vector database

Perform semantic retrieval and LLM-based answering

Export analysis results to CSV, PDF, and JSON

Support for interactive CLI and browser-based UI

3. System Components
3.1 RAG.py
Core document processing and retrieval pipeline

URL validation, PDF downloading, text chunking, embedding, and FAISS storage

QA chain setup with Groq LLM integration

Batch and single-question answering

Metadata storage and retrieval for loaded documents

3.2 rag_question_cli.py
Interactive and direct-mode CLI interface

Automatic intelligent question generation

Full terminal output formatting

CSV/PDF export functionality

3.3 web_interface.py
Streamlit-based web interface

PDF URL input or file upload support

Real-time progress display

Interactive question-answer viewing

Download buttons for CSV, PDF, JSON

4. Installation Procedure
4.1 Clone Repository
bash
Copy code
git clone https://github.com/USERNAME/REPO.git
cd REPO
4.2 Install Dependencies
bash
Copy code
pip install -r requirements.txt
4.3 Environment Setup
Create a .env file in the root directory:

env
Copy code
GROQ_API_KEY=your_groq_api_key
4.4 Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
# Activate:
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
5. Usage
5.1 Command-Line Interface
bash
Copy code
python rag_question_cli.py
Supports both interactive mode and direct mode with arguments:

bash
Copy code
python rag_question_cli.py --url "https://example.com/report.pdf" --company "Apple" --topic "sustainability" --count 8 --export both
5.2 Web Interface
bash
Copy code
streamlit run web_interface.py
6. Project Structure
bash
Copy code
.
├── RAG.py                 # Core backend with document processing & QA logic
├── rag_question_cli.py    # CLI interface
├── web_interface.py       # Web UI with Streamlit
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── utils/                 # Helper utilities
├── results/               # Generated analysis exports
├── faiss_vectorstore/     # Persistent FAISS storage
└── README.md              # Documentation
7. Dependencies
Key packages from requirements.txt:

LangChain (langchain, langchain-community, langchain-groq)

Vector store: faiss-cpu

Embeddings: sentence-transformers

LLM API: groq

PDF parsing: PyMuPDF

Web UI: streamlit

Export: reportlab

Environment management: python-dotenv, pydantic-settings

8. Licensing
This project is licensed under the MIT License, allowing open usage, modification, and distribution.