#!/usr/bin/env python3
"""
Enhanced RAG Question CLI with URL Support
Supports analyzing PDFs from internet URLs with intelligent question generation
"""

import argparse
import os
import csv
import sys
import textwrap
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Optional
import requests
import logging

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
RAG_AVAILABLE = False
try:
    from RAG import DocumentQA, settings
    from langchain_groq import ChatGroq
    RAG_AVAILABLE = True
    logger.info("✅ RAG system available")
except ImportError as e:
    logger.error(f"❌ Import Error: {e}")
    logger.info("💡 Make sure all requirements are installed: pip install -r requirements.txt")

class EnhancedRAGAnalyzer:
    def __init__(self):
        if not RAG_AVAILABLE:
            raise ImportError("RAG system not available. Please check your installation.")
        
        self.rag_qa = DocumentQA()
        self.company_name = ""
        self.pdf_source = ""
        
    def is_valid_url(self, url: str) -> bool:
        """Validate if the provided string is a valid URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file"""
        try:
            if url.lower().endswith('.pdf'):
                return True
            
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'pdf' in content_type
        except:
            return False
    
    def get_company_from_url(self, url: str) -> str:
        """Extract potential company name from URL"""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '').replace('.com', '').replace('.org', '').replace('.net', '')
            company = domain.split('.')[0]
            return company.capitalize()
        except:
            return "Unknown Company"
    
    def interactive_pdf_input(self) -> tuple:
        """Interactive method to get PDF URL and company info"""
        print("\n" + "="*80)
        print("🤖 RAG PDF ANALYZER - Enhanced Version")
        print("="*80)
        print("📋 This tool analyzes company reports from PDF URLs and generates")
        print("   intelligent questions with comprehensive answers.")
        print("-"*80)
        
        # Get PDF URL
        while True:
            print("\n📎 STEP 1: PDF Source")
            pdf_url = input("Enter the PDF URL to analyze: ").strip()
            
            if not pdf_url:
                print("❌ Please enter a valid URL")
                continue
                
            if not self.is_valid_url(pdf_url):
                print("❌ Invalid URL format. Please enter a complete URL (e.g., https://example.com/report.pdf)")
                continue
            
            print(f"🔍 Checking PDF availability...")
            if not self.is_pdf_url(pdf_url):
                response = input("⚠️  URL may not be a PDF. Continue anyway? (y/n): ").lower()
                if response != 'y':
                    continue
            
            break
        
        # Get company name
        print(f"\n🏢 STEP 2: Company Information")
        suggested_company = self.get_company_from_url(pdf_url)
        print(f"   Detected company from URL: {suggested_company}")
        
        company_input = input(f"Enter company name (press Enter for '{suggested_company}'): ").strip()
        company_name = company_input if company_input else suggested_company
        
        # Get analysis focus/topic
        print(f"\n🎯 STEP 3: Analysis Focus")
        print("   Examples: sustainability, financial performance, ESG, innovation, etc.")
        topic = input("Enter analysis topic/focus (optional): ").strip()
        if not topic:
            topic = "general business analysis"
        
        # Get question count
        print(f"\n📊 STEP 4: Analysis Depth")
        max_questions = settings.max_questions if settings else 15
        while True:
            try:
                count_input = input(f"Number of questions to generate (default: 5, max: {max_questions}): ").strip()
                question_count = int(count_input) if count_input else 5
                if 1 <= question_count <= max_questions:
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {max_questions}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        self.company_name = company_name
        self.pdf_source = pdf_url
        
        return pdf_url, company_name, topic, question_count
    
    def load_pdf_from_url(self, url: str) -> bool:
        """Load and process PDF from URL with progress tracking"""
        try:
            print(f"\n🔄 Loading PDF from URL...")
            print(f"   Source: {url}")
            
            # Progress callback
            def progress_callback(message: str, progress: float):
                print(f"   {message} ({progress*100:.0f}%)")
            
            # Use the existing RAG system to load PDF
            self.rag_qa.load_pdf_url(url, progress_callback=progress_callback)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF: {e}")
            print(f"❌ Failed to load PDF: {str(e)}")
            return False
    
    def generate_intelligent_questions(self, company: str, topic: str, count: int = 5) -> List[str]:
        """Generate intelligent questions based on document content"""
        try:
            if not RAG_AVAILABLE:
                return self._fallback_questions(company, topic, count)
            
            # Enhanced prompt for better question generation
            prompt = f"""You are analyzing a {company} company report focused on {topic}. 
            Based on the document content you have access to, generate {count} highly specific, insightful questions that:
            
            1. Target the most important and impactful information in the document
            2. Focus on quantifiable metrics, specific initiatives, and concrete outcomes
            3. Address strategic decisions, performance indicators, and future plans
            4. Are directly answerable from the document content
            5. Provide maximum value for understanding {company}'s {topic}
            
            Make questions detailed and professional. Focus on:
            - Financial metrics and performance indicators
            - Strategic initiatives and their outcomes  
            - Risk factors and mitigation strategies
            - Future outlook and growth plans
            - Competitive positioning and market analysis
            - Specific targets, timelines, and commitments
            
            Return ONLY the questions, one per line, without numbering."""
            
            try:
                llm = ChatGroq(
                    temperature=0.2,
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile"
                )
                
                response = llm.invoke(prompt)
                text = response.content if hasattr(response, 'content') else str(response)
                
                # Clean and extract questions
                questions = []
                for line in text.split('\n'):
                    clean_line = line.strip()
                    if clean_line and '?' in clean_line:
                        # Remove numbering and bullet points
                        clean_line = clean_line.lstrip('0123456789.- ')
                        if clean_line.endswith('?'):
                            questions.append(clean_line)
                            if len(questions) >= count:
                                break
                
                # If we got good questions, return them
                if len(questions) >= max(1, count // 2):  # At least half or 1 question
                    # Pad with fallback if needed
                    if len(questions) < count:
                        fallback = self._fallback_questions(company, topic, count - len(questions))
                        questions.extend(fallback)
                    return questions[:count]
                else:
                    # Fall back to generated questions
                    return self._fallback_questions(company, topic, count)
                    
            except Exception as e:
                logger.warning(f"LLM question generation failed: {e}")
                return self._fallback_questions(company, topic, count)
            
        except Exception as e:
            logger.error(f"❌ Error generating questions: {e}")
            return self._fallback_questions(company, topic, count)
    
    def _fallback_questions(self, company: str, topic: str, count: int) -> List[str]:
        """Generate fallback questions when LLM fails"""
        base_questions = [
            f"What are {company}'s primary strategic objectives and key performance indicators related to {topic}?",
            f"What specific targets, timelines, and commitments has {company} set for {topic}?",
            f"What are the main risks, challenges, and mitigation strategies {company} identifies for {topic}?",
            f"What specific initiatives, programs, or investments has {company} implemented regarding {topic}?",
            f"How does {company} measure, track, and report progress and success in {topic}?",
            f"What are {company}'s future plans, roadmap, and long-term vision for {topic}?",
            f"How does {company}'s performance and approach in {topic} compare to industry standards or competitors?",
            f"What resources, budget allocation, or organizational changes has {company} dedicated to {topic}?",
            f"What are the key achievements, milestones, and outcomes {company} reports for {topic}?",
            f"How does {company} engage with stakeholders, partners, or customers regarding {topic}?",
            f"What regulatory compliance, standards, or certifications does {company} maintain for {topic}?",
            f"What innovation, technology, or methodology advances has {company} made in {topic}?"
        ]
        
        return base_questions[:count]
    
    def format_terminal_output(self, results: List[tuple], company: str, topic: str):
        """Display full results in terminal without truncation"""
        print("\n" + "="*120)
        print(f"📊 COMPREHENSIVE ANALYSIS RESULTS")
        print(f"🏢 Company: {company}")
        print(f"🎯 Topic: {topic}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 Source: {self.pdf_source}")
        print("="*120)
        
        for i, (q_num, question, answer) in enumerate(results):
            print(f"\n📝 QUESTION {q_num}:")
            print("-" * 80)
            
            # Wrap question text
            wrapped_question = textwrap.fill(question, width=116, initial_indent="Q: ", subsequent_indent="   ")
            print(wrapped_question)
            
            print("\n💡 ANSWER:")
            print("-" * 80)
            
            # Wrap answer text with proper indentation
            wrapped_answer = textwrap.fill(answer, width=116, initial_indent="A: ", subsequent_indent="   ")
            print(wrapped_answer)
            
            if i < len(results) - 1:  # Don't print separator after last item
                print("\n" + "─" * 120)
        
        print("\n" + "="*120)
        print("✅ ANALYSIS COMPLETE")
        print("="*120)
    
    def export_to_csv(self, results: List[tuple], company: str, topic: str, filename: Optional[str] = None) -> str:
        """Export results to well-formatted CSV"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_company = "".join(c for c in company if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"results/{safe_company}_{safe_topic}_{timestamp}.csv"
            
            os.makedirs("results", exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Metadata
                writer.writerow(["RAG PDF ANALYSIS REPORT"])
                writer.writerow(["Company", company])
                writer.writerow(["Topic", topic])
                writer.writerow(["Source URL", self.pdf_source])
                writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Data
                writer.writerow(["Question Number", "Question", "Answer"])
                for result in results:
                    writer.writerow(result)
            
            print(f"📄 CSV exported: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            return ""
    
    def export_to_pdf(self, results: List[tuple], company: str, topic: str, filename: Optional[str] = None) -> str:
        """Export results to well-formatted PDF"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_company = "".join(c for c in company if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"results/{safe_company}_{safe_topic}_{timestamp}.pdf"
            
            os.makedirs("results", exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=1*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=20,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            header_style = ParagraphStyle(
                'HeaderStyle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                textColor=colors.darkslategray
            )
            
            question_style = ParagraphStyle(
                'QuestionStyle',
                parent=styles['Normal'],
                fontSize=12,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.darkblue,
                leftIndent=20
            )
            
            answer_style = ParagraphStyle(
                'AnswerStyle',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=15,
                leftIndent=20,
                rightIndent=20
            )
            
            # Title and metadata
            story.append(Paragraph("RAG PDF ANALYSIS REPORT", title_style))
            story.append(Spacer(1, 20))
            
            story.append(Paragraph(f"<b>Company:</b> {company}", header_style))
            story.append(Paragraph(f"<b>Analysis Topic:</b> {topic}", header_style))
            story.append(Paragraph(f"<b>Source URL:</b> {self.pdf_source}", header_style))
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", header_style))
            story.append(Spacer(1, 30))
            
            # Questions and answers
            for q_num, question, answer in results:
                story.append(Paragraph(f"<b>Question {q_num}:</b> {question}", question_style))
                story.append(Paragraph(f"<b>Answer:</b> {answer}", answer_style))
                story.append(Spacer(1, 10))
            
            doc.build(story)
            print(f"📄 PDF exported: {filename}")
            return filename
            
        except ImportError:
            print("❌ PDF export requires reportlab. Install with: pip install reportlab")
            return ""
        except Exception as e:
            logger.error(f"❌ PDF export failed: {e}")
            return ""

def main():
    # Check if RAG system is available
    if not RAG_AVAILABLE:
        print("❌ RAG system is not available. Please install required dependencies:")
        print("   pip install -r requirements.txt")
        print("   Make sure you have a .env file with GROQ_API_KEY")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="🤖 Enhanced RAG PDF Analyzer - Analyze any PDF from the internet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_question_cli.py
    # Interactive mode - prompts for PDF URL and analysis parameters
  
  python rag_question_cli.py --url "https://example.com/report.pdf" --company "Apple" --topic "sustainability"
    # Direct analysis mode
  
  python rag_question_cli.py --url "https://example.com/report.pdf" --count 8 --export both
    # Generate 8 questions and export to both CSV and PDF
        """
    )
    
    parser.add_argument("--url", help="PDF URL to analyze")
    parser.add_argument("--company", help="Company name")  
    parser.add_argument("--topic", help="Analysis topic/focus")
    parser.add_argument("--count", type=int, default=5, help="Number of questions (1-15)")
    parser.add_argument("--export", choices=['csv', 'pdf', 'both'], help="Export results")
    parser.add_argument("--output", help="Custom output filename prefix")
    
    args = parser.parse_args()

    try:
        analyzer = EnhancedRAGAnalyzer()
        
        # Interactive or direct mode
        if args.url and args.company and args.topic:
            # Direct mode with arguments
            pdf_url = args.url
            company = args.company
            topic = args.topic
            max_questions = settings.max_questions if settings else 15
            question_count = min(max(args.count, 1), max_questions)
        else:
            # Interactive mode
            pdf_url, company, topic, question_count = analyzer.interactive_pdf_input()
        
        # Load and process PDF
        print(f"\n🚀 Starting analysis...")
        if not analyzer.load_pdf_from_url(pdf_url):
            print("❌ Analysis failed - could not load PDF")
            return
        
        print(f"🧠 Generating {question_count} intelligent questions...")
        questions = analyzer.generate_intelligent_questions(company, topic, question_count)
        
        print(f"💬 Processing questions and generating comprehensive answers...")
        
        # Progress callback for batch processing
        def progress_callback(current: int, total: int):
            print(f"   📝 Processing question {current}/{total}...")
        
        # Use batch processing for better performance
        results = analyzer.rag_qa.ask_batch(questions, progress_callback)

        # Display full results in terminal
        analyzer.format_terminal_output(results, company, topic)
        
        # Export functionality
        if args.export or len(sys.argv) == 1:  # Always offer export in interactive mode
            if not args.export:
                export_choice = input("\n📤 Export results? (csv/pdf/both/n): ").lower()
                args.export = export_choice if export_choice in ['csv', 'pdf', 'both'] else None
            
            if args.export:
                print(f"\n📤 Exporting results...")
                
                if args.export in ['csv', 'both']:
                    csv_filename = None
                    if args.output:
                        csv_filename = f"{args.output}.csv"
                    analyzer.export_to_csv(results, company, topic, csv_filename)
                
                if args.export in ['pdf', 'both']:
                    pdf_filename = None
                    if args.output:
                        pdf_filename = f"{args.output}.pdf"
                    analyzer.export_to_pdf(results, company, topic, pdf_filename)
                
                print(f"📁 Files saved in: results/ directory")
        
        print(f"\n✅ Analysis complete! Processed {len(results)} questions successfully.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   1. Check your internet connection")
        print(f"   2. Verify the PDF URL is accessible")  
        print(f"   3. Ensure GROQ_API_KEY is set in your .env file")
        print(f"   4. For PDF export: pip install reportlab")
        print(f"   5. Try with a smaller PDF or fewer questions")

if __name__ == "__main__":
    main()