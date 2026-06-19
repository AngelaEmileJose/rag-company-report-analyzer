import os
import argparse
import asyncio
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# LangChain / FAISS imports (reusing embedding and vector store approach)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from retrieval_plugin import RetrievalPlugin

# Load environment variables
load_dotenv()

def extract_documents_from_pdf(pdf_path: str) -> list[Document]:
    """Extract text from a PDF and return LangChain Documents with page metadata.

    Each Document's metadata['page'] is the 1-indexed page number, which the
    retrieval plugin uses to produce [Chunk N | Page M | score X] context blocks.
    """
    print(f"Reading PDF: {pdf_path}...")
    doc = fitz.open(pdf_path)
    documents = []
    for page_idx, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_idx}))
    print(f"Extracted {len(documents)} non-empty pages.")
    return documents

def build_vector_store(documents: list[Document]):
    """Chunk Documents and build a FAISS vector index using SentenceTransformers embeddings.

    Page metadata is preserved on each chunk so the retrieval plugin can cite sources.
    """
    print("Chunking documents and creating vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    # split_documents preserves Document + metadata
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # Use the same model as the parent project
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")
    return vector_store

async def run_semantic_kernel_pipeline(pdf_path: str, model_name: str, endpoint: str):
    """Set up and run the Semantic Kernel RAG pipeline."""
    # 1. Ingest PDF and index it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    documents = extract_documents_from_pdf(pdf_path)
    vector_store = build_vector_store(documents)
    
    # 2. Initialize Semantic Kernel
    print("\nInitializing Semantic Kernel...")
    kernel = Kernel()
    
    # 3. Add Local Ollama Chat Completion Service
    # Ollama offers an OpenAI-compatible endpoint at /v1. We point the OpenAIChatCompletion connector to it.
    # Note: Semantic Kernel also supports direct OllamaChatCompletion connector, but using the OpenAI-compatible
    # endpoint is the most standard and reliable pattern across different SDK versions.
    from openai import AsyncOpenAI
    
    ollama_endpoint = f"{endpoint.rstrip('/')}/v1"
    print(f"Connecting to Ollama model '{model_name}' at endpoint: {ollama_endpoint}")
    
    custom_client = AsyncOpenAI(
        api_key="ollama-is-local",
        base_url=ollama_endpoint
    )
    
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="local_ollama",
            ai_model_id=model_name,
            async_client=custom_client
        )
    )
    
    # 4. Register Native Retrieval Plugin
    print("Registering native RetrievalPlugin...")
    retrieval_plugin = RetrievalPlugin(vector_store=vector_store, retrieval_k=5)
    kernel.add_plugin(retrieval_plugin, plugin_name="RetrievalPlugin")
    
    # 5. Register Semantic Plugins (Skills) from directories
    print("Loading semantic plugins from './plugins'...")
    plugins_directory = os.path.join(os.path.dirname(__file__), "plugins")
    report_plugin = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="ReportPlugin")
    
    # 6. Execute Skills
    # First, let's query the vector store via the native plugin to retrieve context
    query = "What is the financial performance, revenue, profitability, and operational risks of the company?"
    print(f"\nExecuting Native Retrieval for query: '{query}'")
    
    retrieve_func = kernel.get_function(plugin_name="RetrievalPlugin", function_name="RetrieveContext")
    context_result = await kernel.invoke(retrieve_func, query=query)
    context_str = str(context_result)
    
    print(f"Retrieved context length: {len(context_str)} characters.")
    
    # Run Skill 1: SummarizeReport
    print("\n--- Running SummarizeReport Skill ---")
    summarize_func = kernel.get_function(plugin_name="ReportPlugin", function_name="SummarizeReport")
    summary = await kernel.invoke(summarize_func, input=context_str)
    print("\nSummary Result:")
    print("=" * 60)
    print(summary)
    print("=" * 60)
    
    # Run Skill 2: ExtractKeyMetrics
    print("\n--- Running ExtractKeyMetrics Skill ---")
    extract_func = kernel.get_function(plugin_name="ReportPlugin", function_name="ExtractKeyMetrics")
    metrics = await kernel.invoke(extract_func, input=context_str)
    print("\nExtracted Metrics:")
    print("=" * 60)
    print(metrics)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Analyzer with Semantic Kernel and Ollama.")
    parser.add_argument(
        "--pdf_path", 
        type=str, 
        default="../data/uploads/TSLA-Q3-2024-Update.pdf",
        help="Path to the PDF document to analyze."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="hermes3:8b",
        help="Ollama model name (e.g. hermes3:8b, gemma2:9b, llama3)."
    )
    parser.add_argument(
        "--endpoint", 
        type=str, 
        default="http://localhost:11434",
        help="Ollama API base URL."
    )
    
    args = parser.parse_args()
    
    # Run async main loop
    asyncio.run(run_semantic_kernel_pipeline(
        pdf_path=args.pdf_path,
        model_name=args.model,
        endpoint=args.endpoint
    ))
