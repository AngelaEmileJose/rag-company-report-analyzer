import logging
import hashlib
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Core dependencies
import numpy as np
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# ReportLab imports for PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.colors import black, blue, grey, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from .config import RAGConfig
from .document_processor import DocumentProcessor
from .models import ProcessingResult
from .utils import setup_logging

class ImprovedRAGSystem:
    """Enhanced RAG system with modular design"""

    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Setup logging
        setup_logging(
            log_level=config.log_level,
            log_to_file=config.log_to_file,
            logs_dir=config.logs_dir,
            max_log_size_mb=config.max_log_size_mb
        )
        self.logger = logging.getLogger("rag_system.core")

        # Initialize components
        self.document_processor = DocumentProcessor(config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )

        # Initialize LLM with retry logic
        self.llm = ChatGroq(
            groq_api_key=config.groq_api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries
        )

        # Vector store
        self.current_vector_store: Optional[FAISS] = None
        self.current_source: Optional[str] = None
    
    async def process_document(self, source: str) -> ProcessingResult:
        """Process a document and create vector store"""
        start_time = datetime.now()

        try:
            self.logger.info(f"Processing document: {source}")

            # Extract text
            text, processed_source = await self.document_processor.process_source(source)

            if not text or len(text.strip()) < 100:
                return ProcessingResult(
                    success=False,
                    message="Document appears to be empty or too short",
                    source=source
                )

            self.logger.info(f"Extracted {len(text)} characters from document")

            # Create chunks
            documents = [Document(page_content=text, metadata={"source": processed_source})]
            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                return ProcessingResult(
                    success=False,
                    message="Failed to create text chunks",
                    source=source
                )

            self.logger.info(f"Created {len(chunks)} chunks")

            # Create vector store
            try:
                self.current_vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.current_source = processed_source
                self.logger.info("Vector store created successfully")
            except Exception as e:
                self.logger.error(f"Vector store creation failed: {e}")
                return ProcessingResult(
                    success=False,
                    message=f"Vector store creation failed: {str(e)}",
                    source=source
                )

            # Cache vector store
            if self.config.enable_cache:
                try:
                    cache_key = hashlib.md5(source.encode()).hexdigest()
                    vector_store_path = self.config.vector_store_dir / f"vector_store_{cache_key}"
                    self.current_vector_store.save_local(str(vector_store_path))
                    self.logger.info(f"Cached vector store to {vector_store_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache vector store: {e}")

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                success=True,
                message="Document processed successfully",
                chunk_count=len(chunks),
                processing_time=processing_time,
                source=processed_source
            )

        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Processing failed: {str(e)}",
                source=source
            )

    def ask_question(self, question: str) -> str:
        """Ask a question using the current vector store"""
        if not self.current_vector_store:
            return "Error: No document loaded. Please process a document first."

        try:
            self.logger.info(f"Processing question: {question}")

            # Retrieve relevant documents
            relevant_docs = []
            
            # Use safe retrieval logic (trying multiple k values if needed)
            try:
                relevant_docs = self.current_vector_store.similarity_search(
                    question,
                    k=self.config.retrieval_k
                )
            except Exception as e:
                self.logger.warning(f"Initial retrieval failed: {e}")

            if not relevant_docs:
                return "I couldn't find relevant information to answer your question. The document may not contain information about this topic, or there might be an issue with the search system."

            # Create context
            context_parts = []
            total_length = 0
            max_context_length = 8000 

            for doc in relevant_docs:
                if total_length + len(doc.page_content) > max_context_length:
                    break
                context_parts.append(doc.page_content)
                total_length += len(doc.page_content)

            context = "\n\n".join(context_parts)

            # Generate answer
            prompt = f"""Based on the following context from the document, answer the question accurately and concisely. If the answer is not fully contained in the context, indicate what information is available and what might be missing.

Context:
{context}

Question: {question}

Please provide a clear, informative answer based on the available information:"""

            try:
                response = self.llm.invoke(prompt)
                answer = getattr(response, "content", str(response))
                return answer

            except Exception as e:
                self.logger.error(f"LLM invocation failed: {e}")
                return f"I found relevant information but encountered an error generating the response: {str(e)}"

        except Exception as e:
            self.logger.error(f"Question answering failed: {str(e)}")
            return f"Error processing question: {str(e)}"

    def generate_executive_summary(self) -> List[Dict[str, str]]:
        """Generate a comprehensive executive summary via Q&A"""
        if not self.current_vector_store:
            return [{"question": "Error", "answer": "No document loaded."}]

        strategic_questions = [
            ("Financial Performance", "What are the key financial highlights, including revenue, profit, and margins?"),
            ("Operational Updates", "What are the key operational milestones, production numbers, or delivery figures?"),
            ("Future Outlook", "What is the company's future outlook, guidance, or major upcoming product launches?"),
            ("Risks & Challenges", "What are the main risks, challenges, or headwinds mentioned?"),
            ("Strategic Initiatives", "What are the key strategic initiatives or technology updates discussed?")
        ]

        summary = []
        self.logger.info("Generating executive summary...")

        for topic, question in strategic_questions:
            try:
                answer = self.ask_question(question)
                summary.append({
                    "topic": topic,
                    "question": question,
                    "answer": answer
                })
            except Exception as e:
                self.logger.error(f"Failed to answer summary question '{topic}': {e}")

        return summary

    def generate_questions(self, company_name: str, topic: str, count: int = 5) -> List[str]:
        """Generate intelligent questions about the document"""
        if not self.current_vector_store:
            self.logger.error("No document loaded for question generation")
            return []

        try:
            # Get sample content
            sample_docs = []
            
            # Improve search strategy:
            # 1. Search for the specific topic + company
            # 2. Search for the topic alone
            # 3. Search for broad "key highlights" type terms if topic is generic
            search_terms = [
                f"{company_name} {topic}", 
                topic, 
                f"{topic} highlights",
                f"{topic} analysis",
                "executive summary", 
                "key financial results",
                "risk factors"
            ]

            # Collect more distinctive chunks
            seen_content = set()
            
            for term in search_terms:
                if len(sample_docs) >= 8: # Increased from 6
                    break
                try:
                    docs = self.current_vector_store.similarity_search(term, k=3)
                    for doc in docs:
                        # Deduplicate based on content hash or start of string
                        content_snippet = doc.page_content[:100]
                        if content_snippet not in seen_content:
                            seen_content.add(content_snippet)
                            sample_docs.append(doc)
                            if len(sample_docs) >= 8:
                                break
                except Exception:
                    continue

            if not sample_docs:
                # If search fails to find anything (rare), try to get *any* documents (e.g. from the store directly if possible, or just fail)
                # For FAISS, we can't easily iterate all, but we can search for empty string or common words
                try:
                    sample_docs = self.current_vector_store.similarity_search(company_name, k=5)
                except:
                    return []

            context_parts = []
            for doc in sample_docs:
                # Use more context per doc (up to 1500 chars) to give LLM enough meat
                context_parts.append(doc.page_content[:1500])
            context = "\n---\n".join(context_parts)

            prompt = f"""You are an expert financial analyst reviewing a report for {company_name}.
Your task is to generate {count} specific, high-quality questions about "{topic}" based *strictly* on the provided document excerpts.

CRITICAL INSTRUCTIONS:
1. Questions MUST reference specific details, numbers, dates, or entities found in the text.
2. AVOID generic questions (e.g., "What are the risks?" or "How is the revenue?"). 
   Instead ask: "How did the supply chain disruption in Q3 affect the gross margin?" (if mentioned).
3. If the text mentions specific project names, product lines, or competitors, include them in the questions.
4. Ensure the questions are diverse and cover different aspects of the topic.
5. The questions should be answerable using facts from the document.

Document Excerpts:
{context}

Generate exactly {count} specific questions:"""

            try:
                response = self.llm.invoke(prompt)
                lines = getattr(response, "content", str(response)).split('\n')
                questions = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Clean numbering and bullets
                    q = re.sub(r'^(\d+\.|[-•*])\s*', '', line).strip()
                    
                    # Filter for valid question format
                    if (len(q) > 20 and 
                        q.endswith('?') and 
                        q not in questions):
                        questions.append(q)
                        if len(questions) >= count:
                            break
                
                # If we didn't get enough valid questions, fill with generic ones (as fallback only)
                if len(questions) < count:
                    placeholders = [
                        f"What are the key takeaways regarding {topic}?",
                        f"How does {company_name} plan to address challenges in {topic}?",
                        f"What are the major metrics reported for {topic}?",
                        f"What strategic initiatives relate to {topic}?",
                        f"How has the performance in {topic} changed year-over-year?"
                    ]
                    for pq in placeholders:
                        if len(questions) >= count:
                            break
                        if pq not in questions:
                            questions.append(pq)
                
                return questions[:count]

            except Exception as e:
                self.logger.error(f"LLM question generation failed: {e}")
                return []

        except Exception as e:
            self.logger.error(f"Question generation failed: {str(e)}")
            return []

    def export_results_to_pdf(self, questions: List[str], answers: List[str],
                            company: str, topic: str, filename: Optional[str] = None) -> str:
        """Export Q&A results to PDF"""
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_clean = company.replace(' ', '_')
            topic_clean = topic.replace(' ', '_')
            filename = f"RAG_Analysis_{company_clean}_{topic_clean}_{timestamp}.pdf"

        filepath = results_dir / filename

        doc = SimpleDocTemplate(str(filepath), pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, spaceAfter=30, textColor=blue, alignment=TA_CENTER)
        section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=14, spaceAfter=12, textColor=black)
        question_style = ParagraphStyle('QuestionStyle', parent=styles['Normal'], fontSize=11, spaceAfter=8, textColor=blue, leftIndent=20)
        answer_style = ParagraphStyle('AnswerStyle', parent=styles['Normal'], fontSize=10, spaceAfter=15, textColor=black, leftIndent=30, alignment=TA_JUSTIFY)

        story = []
        story.append(Paragraph("RAG Document Analysis Report", title_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Analysis Details", section_style))
        source_display = self.current_source or 'Unknown'
        if source_display and len(source_display) > 60:
            source_display = source_display[:60] + "..."
            
        metadata_data = [
            ["Company", company],
            ["Topic", topic],
            ["Source", source_display],
            ["Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Model", self.config.model_name],
            ["Questions", str(len(questions))]
        ]

        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), grey),
            ('TEXTCOLOR', (0, 0), (0, -1), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))

        story.append(metadata_table)
        story.append(Spacer(1, 30))

        story.append(Paragraph("Questions & Answers", section_style))
        story.append(Spacer(1, 15))

        for i, (question, answer) in enumerate(zip(questions, answers), 1):
            clean_question = question.replace('**', '').strip()
            clean_answer = answer.replace('**', '').replace('*', '•').replace('\n\n', '<br/><br/>').replace('\n', ' ')

            story.append(Paragraph(f"Q{i}: {clean_question}", question_style))
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>Answer:</b> {clean_answer}", answer_style))
            story.append(Spacer(1, 20))

        story.append(Spacer(1, 30))
        footer_text = f"Generated by RAG Document Analyzer on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(footer_text, styles['Normal']))

        doc.build(story)
        return str(filepath)
