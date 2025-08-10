# RAG.py - Enhanced version with URL support and improved DocumentQA class
import os
import time
import logging
import requests
import tempfile
from dotenv import load_dotenv
from typing import Union, List, Optional, Callable
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import pickle
from urllib.parse import urlparse
from pydantic_settings import BaseSettings


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for RAG system"""
    groq_api_key: str = ""
    max_pdf_size: int = 50_000_000  # 50MB limit
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_questions: int = 15
    download_timeout: int = 60
    max_retries: int = 3
    
    class Config:
        env_file = ".env"

# Initialize settings
try:
    settings = Settings()
    GROQ_API_KEY = settings.groq_api_key or os.getenv("GROQ_API_KEY")
except Exception as e:
    logger.warning(f"⚠️ Settings initialization failed: {e}")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    settings = None

if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY is not set in .env! Some features may not work.")

class DocumentQA:
    """Enhanced DocumentQA class with URL support and improved functionality"""
    
    def __init__(self):
        """Initialize the DocumentQA system"""
        try:
            self.embedder = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
            self.vectorstore = None
            self.qa_chain = None
            self.vector_store_path = "faiss_vectorstore"
            self.current_document_source = ""
            self.document_metadata = {}
            logger.info("✅ DocumentQA initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing DocumentQA: {e}")
            raise

    def _validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            
            # Quick HEAD request to check if URL is accessible
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"URL validation failed: {e}")
            return False

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file"""
        try:
            if url.lower().endswith('.pdf'):
                return True
            
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'pdf' in content_type
        except:
            return False

    def load_pdf_url(self, url: str, collection_name: str = "documents", 
                     progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        Load PDF from URL and store in FAISS vector database
        
        Args:
            url: Direct URL to PDF file
            collection_name: Name for the document collection
            progress_callback: Optional callback for progress updates (message, progress)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If URL is invalid or PDF cannot be processed
            requests.RequestException: If download fails
        """
        try:
            if progress_callback:
                progress_callback("Validating URL...", 0.1)
            
            if not self._validate_url(url):
                raise ValueError(f"Invalid or inaccessible URL: {url}")
            
            if not self._is_pdf_url(url):
                logger.warning("⚠️ URL may not point to a PDF file")
            
            logger.info(f"📥 Downloading PDF from {url}...")
            if progress_callback:
                progress_callback("Downloading PDF...", 0.2)
            
            # Download PDF with better error handling and progress tracking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, timeout=settings.download_timeout if settings else 60, 
                                  headers=headers, stream=True)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > (settings.max_pdf_size if settings else 50_000_000):
                raise ValueError(f"PDF file too large: {content_length} bytes")
            
            # Create temporary file and download in chunks
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                total_bytes = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        total_bytes += len(chunk)
                temp_pdf_path = temp_file.name
            
            logger.info(f"📄 PDF downloaded successfully ({total_bytes} bytes)")
            if progress_callback:
                progress_callback("Processing PDF...", 0.4)
            
            # Load and split documents
            loader = PyMuPDFLoader(temp_pdf_path)
            documents = loader.load()
            logger.info(f"📄 Loaded {len(documents)} pages from PDF")

            if not documents:
                raise ValueError("No content could be extracted from the PDF")

            # Enhanced text splitting for better chunks
            chunk_size = settings.chunk_size if settings else 1200
            chunk_overlap = settings.chunk_overlap if settings else 200
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len
            )
            docs = splitter.split_documents(documents)
            logger.info(f"🔪 Split into {len(docs)} chunks")

            if not docs:
                raise ValueError("No text chunks could be created from the PDF")

            if progress_callback:
                progress_callback("Creating embeddings...", 0.6)

            # Create embeddings and FAISS vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create FAISS vector store
            logger.info("🔄 Creating FAISS vector store...")
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Save vector store to disk
            os.makedirs(os.path.dirname(self.vector_store_path) or '.', exist_ok=True)
            self.vectorstore.save_local(self.vector_store_path)
            logger.info(f"💾 Vector store saved to {self.vector_store_path}")

            if progress_callback:
                progress_callback("Setting up QA chain...", 0.8)

            # Create QA chain
            self._create_qa_chain()
            
            # Store metadata
            self.current_document_source = url
            self.document_metadata = {
                'source': url,
                'pages': len(documents),
                'chunks': len(docs),
                'size_bytes': total_bytes,
                'collection': collection_name,
                'processed_at': time.time()
            }

            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
            
            if progress_callback:
                progress_callback("Analysis ready!", 1.0)
                
            logger.info("✅ PDF processed and stored in FAISS vector database!")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Error downloading PDF: {e}")
            raise ValueError(f"Failed to download PDF from URL: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error loading PDF: {e}", exc_info=True)
            raise ValueError(f"Failed to process PDF: {str(e)}")

    def load_pdf_file(self, file_path: str, collection_name: str = "documents") -> bool:
        """Load PDF from local file path"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            logger.info(f"📥 Loading PDF from {file_path}...")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > (settings.max_pdf_size if settings else 50_000_000):
                raise ValueError(f"PDF file too large: {file_size} bytes")
            
            # Load and split documents
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"📄 Loaded {len(documents)} pages from PDF")

            chunk_size = settings.chunk_size if settings else 1200
            chunk_overlap = settings.chunk_overlap if settings else 200
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            docs = splitter.split_documents(documents)
            logger.info(f"🔪 Split into {len(docs)} chunks")

            # Create embeddings and FAISS vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            self.vectorstore.save_local(self.vector_store_path)
            
            # Create QA chain
            self._create_qa_chain()
            
            # Store metadata
            self.current_document_source = file_path
            self.document_metadata = {
                'source': file_path,
                'pages': len(documents),
                'chunks': len(docs),
                'size_bytes': file_size,
                'collection': collection_name,
                'processed_at': time.time()
            }
            
            logger.info("✅ PDF processed and stored in FAISS vector database!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF file: {e}")
            raise

    def _create_qa_chain(self):
        """Create the QA chain with LLM"""
        try:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                max_tokens=1000
            )
            
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 6}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever,
                return_source_documents=True,
                chain_type="stuff"
            )
            
            logger.info("✅ QA chain created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating QA chain: {e}")
            raise

    def load_existing_vectorstore(self) -> bool:
        """Load existing vector store from disk with validation"""
        try:
            if os.path.exists(self.vector_store_path):
                logger.info("🔄 Loading existing vector store...")
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Validate vectorstore
                if not hasattr(self.vectorstore, 'index'):
                    raise ValueError("Invalid vectorstore format")
                
                # Create QA chain
                self._create_qa_chain()
                
                logger.info("✅ Loaded existing vector store from disk")
                return True
            else:
                logger.warning("⚠️ No existing vector store found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return False

    def ask(self, question: str) -> str:
        """Ask a question and get an answer from the loaded documents"""
        if not self.qa_chain:
            if not self.load_existing_vectorstore():
                return "❌ Please load a document first using load_pdf_url() or load_pdf_file()."
        
        try:
            # Enhanced question with context
            enhanced_question = f"""Based on the document content, please provide a comprehensive answer to: {question}
            
            Please include specific details, numbers, and examples from the document when available."""
            
            result = self.qa_chain.invoke({"query": enhanced_question})
            
            answer = result.get('result', 'No answer found.')
            
            # Clean up the answer
            if answer.startswith("Based on the document content"):
                parts = answer.split(":", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error during QA: {e}")
            return f"❌ Error processing question: {str(e)}"

    def ask_batch(self, questions: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> List[tuple]:
        """Ask multiple questions efficiently"""
        results = []
        for i, question in enumerate(questions):
            try:
                answer = self.ask(question)
                results.append((i + 1, question, answer))
                
                if progress_callback:
                    progress_callback(i + 1, len(questions))
                    
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                results.append((i + 1, question, f"Error processing question: {str(e)}"))
        
        return results

    def get_document_info(self) -> dict:
        """Get information about the currently loaded document"""
        return {
            "source": self.current_document_source,
            "vectorstore_loaded": self.vectorstore is not None,
            "qa_chain_ready": self.qa_chain is not None,
            "vector_store_path": self.vector_store_path,
            "metadata": self.document_metadata
        }

    def clear_vectorstore(self):
        """Clear the current vector store"""
        try:
            self.vectorstore = None
            self.qa_chain = None
            self.current_document_source = ""
            self.document_metadata = {}
            
            # Remove saved vector store files
            if os.path.exists(self.vector_store_path):
                import shutil
                shutil.rmtree(self.vector_store_path)
            
            logger.info("✅ Vector store cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing vector store: {e}")

    def get_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Get similar document chunks for a query"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            return []

# Backwards compatibility - keep the original function names
def load_pdf_url(url: str):
    """Backwards compatibility function"""
    qa = DocumentQA()
    qa.load_pdf_url(url)
    return qa

if __name__ == "__main__":
    logger.info("🤖 Testing Enhanced DocumentQA...")
    
    try:
        # Test initialization
        rag = DocumentQA()
        logger.info("✅ DocumentQA initialized successfully!")
        
        # Test with Apple's environmental report (if GROQ_API_KEY is available)
        if GROQ_API_KEY:
            test_url = "https://www.apple.com/environment/pdf/Apple_Environmental_Progress_Report_2024.pdf"
            logger.info(f"🔄 Testing with: {test_url}")
            
            # Progress callback for testing
            def progress_callback(message, progress):
                print(f"Progress: {message} ({progress*100:.0f}%)")
            
            rag.load_pdf_url(test_url, progress_callback=progress_callback)
            
            # Test a question
            answer = rag.ask("What are Apple's main sustainability goals?")
            logger.info(f"🤖 Test Answer: {answer[:200]}...")
            
            # Show document info
            info = rag.get_document_info()
            logger.info(f"📊 Document Info: {info}")
            
        else:
            logger.warning("⚠️ Skipping URL test - GROQ_API_KEY not available")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.info("💡 Make sure GROQ_API_KEY is set in your .env file")