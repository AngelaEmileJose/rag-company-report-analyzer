# RAG Company Report Analyzer

Analyze company reports from PDF URLs or uploads using LangChain, FAISS, and Groq LLMs.  
Includes both a CLI (`rag_question_cli.py`) and a web interface (`web_interface.py` with Streamlit).

## Features
- Load PDF from URL or file
- Generate intelligent, context-aware questions
- Retrieve detailed answers using vector search + LLM
- Export results to CSV, PDF, or JSON

## Installation
```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
