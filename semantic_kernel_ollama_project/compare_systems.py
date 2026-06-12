import os
import sys
import time
import argparse
import asyncio
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv

# Adjust path to import from parent directory (existing RAG system)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import existing RAG classes
try:
    from rag_system import ImprovedRAGSystem, RAGConfig
except ImportError:
    print("Could not import existing RAG system. Make sure you run from the project directories.")
    sys.exit(1)

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval_plugin import RetrievalPlugin

# Load env from parent directory
load_dotenv(dotenv_path=project_root / ".env")

async def run_groq_benchmark(pdf_path: str, query: str):
    """Benchmark the existing LangChain + Groq RAG system."""
    print("\n--- Benchmarking Groq + LangChain System ---")
    
    # Check if Groq key exists
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "gsk_your_groq_api_key_here":
        print("Groq API Key is not set or placeholder. Skipping Groq benchmark.")
        return None
    
    try:
        config = RAGConfig.from_env()
        # Ensure we are using the correct PDF path
        config.data_dir = pdf_path
        
        start_time = time.time()
        
        # Initialize
        rag = ImprovedRAGSystem(config)
        init_time = time.time() - start_time
        print(f"Groq RAG System Initialized in {init_time:.2f}s")
        
        # Process document
        start_proc = time.time()
        proc_result = await rag.process_document(pdf_path)
        proc_time = time.time() - start_proc
        
        if not proc_result.success:
            print(f"Error processing PDF with Groq system: {proc_result.message}")
            return None
            
        print(f"Processed PDF with Groq system in {proc_time:.2f}s")
        
        # Query: Ask question
        start_query = time.time()
        answer = rag.ask_question(query)
        query_time = time.time() - start_query
        print(f"Groq Answer generated in {query_time:.2f}s")
        
        # Summary
        start_summary = time.time()
        summary_results = rag.generate_executive_summary()
        summary_time = time.time() - start_summary
        print(f"Groq Summary generated in {summary_time:.2f}s")
        
        # Combine summary answers
        summary_text = "\n\n".join([f"### {item['topic']}\n{item['answer']}" for item in summary_results])
        
        return {
            "init_time": init_time,
            "processing_time": proc_time,
            "query_time": query_time,
            "summary_time": summary_time,
            "answer": answer,
            "summary": summary_text,
            "model": config.model_name
        }
    except Exception as e:
        print(f"Error running Groq benchmark: {e}")
        return None

async def run_ollama_benchmark(pdf_path: str, model_name: str, endpoint: str, query: str):
    """Benchmark the new Semantic Kernel + Ollama RAG system."""
    print(f"\n--- Benchmarking Ollama + Semantic Kernel ({model_name}) ---")
    
    start_time = time.time()
    
    # 1. Ingest PDF and build FAISS
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    proc_time = time.time() - start_time
    print(f"Local indexing completed in {proc_time:.2f}s")
    
    # 2. Setup Kernel
    kernel = Kernel()
    from openai import AsyncOpenAI
    
    ollama_endpoint = f"{endpoint.rstrip('/')}/v1"
    
    custom_client = AsyncOpenAI(
        api_key="ollama",
        base_url=ollama_endpoint
    )
    
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="local_ollama",
            ai_model_id=model_name,
            async_client=custom_client
        )
    )
    
    retrieval_plugin = RetrievalPlugin(vector_store=vector_store, retrieval_k=5)
    kernel.add_plugin(retrieval_plugin, plugin_name="RetrievalPlugin")
    
    plugins_directory = os.path.join(os.path.dirname(__file__), "plugins")
    report_plugin = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="ReportPlugin")
    
    # 3. Retrieve Context
    retrieve_func = kernel.get_function(plugin_name="RetrievalPlugin", function_name="RetrieveContext")
    context_result = await kernel.invoke(retrieve_func, query=query)
    context_str = str(context_result)
    
    # 4. Run SummarizeReport
    start_summary = time.time()
    summarize_func = kernel.get_function(plugin_name="ReportPlugin", function_name="SummarizeReport")
    summary = await kernel.invoke(summarize_func, input=context_str)
    summary_time = time.time() - start_summary
    print(f"Ollama Summary generated in {summary_time:.2f}s")
    
    # 5. Run ExtractKeyMetrics
    start_metrics = time.time()
    extract_func = kernel.get_function(plugin_name="ReportPlugin", function_name="ExtractKeyMetrics")
    metrics = await kernel.invoke(extract_func, input=context_str)
    metrics_time = time.time() - start_metrics
    print(f"Ollama Metrics generated in {metrics_time:.2f}s")
    
    return {
        "processing_time": proc_time,
        "summary_time": summary_time,
        "metrics_time": metrics_time,
        "summary": str(summary),
        "metrics": str(metrics),
        "model": model_name
    }

def write_report(groq_results, ollama_results, query):
    """Write the comparison report in markdown format."""
    report_path = Path(__file__).resolve().parent / "comparison_report.md"
    
    print(f"\nWriting evaluation report to {report_path}...")
    
    has_groq = groq_results is not None
    
    markdown_content = f"""# RAG Evaluation Report: Groq + LangChain vs. Local Ollama + Semantic Kernel

This report compares the performance and output quality of two Retrieval-Augmented Generation (RAG) system configurations, evaluated using the TSLA Q3 2024 financial update.

## System Configurations

1. **System A (Remote)**:
   - **Framework**: LangChain
   - **LLM Backend**: Groq API
   - **Model**: {groq_results['model'] if has_groq else 'llama-3.3-70b-versatile (Skipped)'}
   
2. **System B (Local)**:
   - **Framework**: Microsoft Semantic Kernel
   - **LLM Backend**: Ollama Local Server
   - **Model**: {ollama_results['model']}

---

## ⚡ Performance Benchmarks (Latency in Seconds)

| Stage | Groq + LangChain (Remote) | Ollama + Semantic Kernel (Local) | Speedup / Overhead |
| :--- | :---: | :---: | :---: |
| **Ingestion & Indexing** | {f"{groq_results['processing_time']:.2f}s" if has_groq else "N/A"} | {ollama_results['processing_time']:.2f}s | Local FAISS indexing is extremely fast |
| **Summarization / Executive QA** | {f"{groq_results['summary_time']:.2f}s" if has_groq else "N/A"} | {ollama_results['summary_time']:.2f}s | {f"Groq is {ollama_results['summary_time']/groq_results['summary_time']:.1f}x faster" if (has_groq and groq_results['summary_time'] > 0) else "N/A"} |
| **Metrics Extraction** | N/A (Standard QA) | {ollama_results['metrics_time']:.2f}s | Dedicated semantic skill run locally |

*Note: Ingestion latency includes document load, chunking, and embedding generation via SentenceTransformers (both systems use local embeddings).*

---

## 📝 Qualitative Analysis

### 1. Executive Summary Output Comparison

#### Local Ollama ({ollama_results['model']}) Summary
```markdown
{ollama_results['summary']}
```

"""
    if has_groq:
        markdown_content += f"""
#### Groq ({groq_results['model']}) Summary (Q&A Aggregation)
```markdown
{groq_results['summary']}
```
"""

    markdown_content += f"""
### 2. Extracted Metrics Output (Local Ollama)

```markdown
{ollama_results['metrics']}
```

---

## 💡 Key Architectural Insights

1. **Semantic Kernel vs. LangChain**:
   - **Semantic Kernel** focuses heavily on **Plugins and Skills** (both native code and semantic markdown prompts). Registration of functions is explicit, clean, and easily organized in file-based structures (`config.json` and `skprompt.txt`).
   - **LangChain** is built on a highly modular chain-of-thought system, which can feel complex when composing custom logic, but is highly integrated with numerous remote ecosystem tools.
   
2. **Local Ollama (Hermes/Gemma) vs. Groq (Llama-3.3)**:
   - **Cost & Privacy**: Local Ollama runs completely offline at zero cost. Ideal for proprietary financial reports where data leakage is a regulatory concern.
   - **Latency**: Groq uses custom LPU hardware, generating hundreds of tokens per second. Local Ollama's speed depends entirely on local CPU/GPU hardware; on consumer laptops, a 8B/9B model will be slower than Groq, but fully self-contained.
   - **Resource Consumption**: Running a local 8B model (Hermes 3) requires ~8GB of VRAM/RAM. Gemma 2 (9B) requires ~10GB.
   
3. **Hermes 3 Model Suitability**:
   - Hermes 3 is highly fine-tuned for agentic capabilities and following structured prompts (like those in our semantic configs). It handles data extraction and summaries exceptionally well, competing closely in quality with larger API-based models.
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Report written successfully.")

async def main():
    parser = argparse.ArgumentParser(description="Benchmark Groq vs. local Ollama RAG setups.")
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
    
    # Resolve absolute path for PDF
    pdf_path = str((Path(__file__).parent / args.pdf_path).resolve())
    
    query = "What is the financial performance, revenue, profitability, and operational risks of the company?"
    
    print(f"Evaluating systems using document: {pdf_path}")
    
    # Run benchmarks
    groq_results = await run_groq_benchmark(pdf_path, query)
    ollama_results = await run_ollama_benchmark(pdf_path, args.model, args.endpoint, query)
    
    # Write comparison report
    write_report(groq_results, ollama_results, query)

if __name__ == "__main__":
    asyncio.run(main())
