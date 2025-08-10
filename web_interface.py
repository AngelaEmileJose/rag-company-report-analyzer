import streamlit as st
import pandas as pd
import os
import io
import time
from datetime import datetime
from urllib.parse import urlparse
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    from RAG import DocumentQA, settings
    RAG_AVAILABLE = True
except Exception as e:
    DocumentQA = None
    settings = None
    RAG_AVAILABLE = False

st.set_page_config(page_title="Company Report Analyzer", layout="wide", page_icon="📊")

# ---- Styles ----
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---- Helpers ----
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_company_from_source(source: str) -> str:
    try:
        if is_valid_url(source):
            domain = urlparse(source).netloc.lower().replace("www.", "")
            return domain.split('.')[0].capitalize()
        return os.path.splitext(os.path.basename(source))[0].capitalize()
    except:
        return "Unknown"

def export_pdf(results_df: pd.DataFrame, company: str, topic: str, source: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=1, textColor=colors.darkblue)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=12, textColor=colors.darkblue)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=10)

    story.append(Paragraph("Company Report Analysis", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Company:</b> {company}", styles['Normal']))
    story.append(Paragraph(f"<b>Topic:</b> {topic}", styles['Normal']))
    story.append(Paragraph(f"<b>Source:</b> {source}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    for _, row in results_df.iterrows():
        story.append(Paragraph(f"Q{row['Question #']}: {row['Question']}", question_style))
        story.append(Paragraph(f"Answer: {row['Answer']}", answer_style))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()

@st.cache_resource
def get_rag():
    if not RAG_AVAILABLE:
        return None
    try:
        return DocumentQA()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

# ---- UI ----
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">📊 Company Report Analyzer</h1>
    <p style="color: white; text-align: center; margin: 0;">Analyze company reports with AI-powered Q&A</p>
</div>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "company" not in st.session_state:
    st.session_state.company = ""
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "source" not in st.session_state:
    st.session_state.source = ""

with st.sidebar:
    st.header("Settings")
    if not RAG_AVAILABLE:
        st.error("❌ RAG not available")
        st.stop()
    if st.session_state.rag is None:
        st.session_state.rag = get_rag()
    if st.session_state.rag:
        st.success("✅ Ready")
    else:
        st.stop()
    question_count = st.slider("Number of Questions", 3, settings.max_questions if settings else 15, 6)
    show_chunks = st.checkbox("Show Retrieved Chunks", False)

tab1, tab2 = st.tabs(["📎 From PDF URL", "📂 Upload PDF File"])

with tab1:
    pdf_url = st.text_input("Enter PDF URL", placeholder="https://example.com/report.pdf")
with tab2:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

topic = st.text_input("Analysis Topic", value="", placeholder="e.g., sustainability, financial performance")
topic = topic if topic.strip() else "general analysis"

if st.button("🚀 Analyze Report", use_container_width=True):
    if not pdf_url and not uploaded_file:
        st.error("Please provide a PDF URL or upload a file.")
        st.stop()

    # Clear old vector store
    st.session_state.rag.clear_vectorstore()

    if pdf_url:
        st.session_state.source = pdf_url
        st.session_state.company = get_company_from_source(pdf_url)
        st.session_state.rag.load_pdf_url(pdf_url)
    else:
        file_path = os.path.join("temp_upload.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.source = uploaded_file.name
        st.session_state.company = get_company_from_source(uploaded_file.name)
        st.session_state.rag.load_pdf(file_path)

    # Generate questions
    questions = [f"Question {i+1} about {st.session_state.company}'s {topic}" for i in range(question_count)]

    results = []
    progress = st.progress(0)
    for i, q in enumerate(questions, start=1):
        ans = st.session_state.rag.ask(q)
        results.append({"Question #": i, "Question": q, "Answer": ans})
        progress.progress(i / question_count)
        time.sleep(5)  # Avoid hitting rate limits

    st.session_state.results = pd.DataFrame(results)
    st.success("✅ Analysis completed")

if st.session_state.results is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Company", st.session_state.company)
    col2.metric("Topic", topic)
    col3.metric("Questions", len(st.session_state.results))

    for _, row in st.session_state.results.iterrows():
        with st.expander(f"Q{row['Question #']}: {row['Question']}"):
            st.write(f"**Answer:** {row['Answer']}")
            if show_chunks:
                chunks = st.session_state.rag.get_similar_chunks(row['Question'], k=3)
                for idx, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {idx}:** {chunk}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_data = st.session_state.results.to_csv(index=False).encode("utf-8")
    pdf_data = export_pdf(st.session_state.results, st.session_state.company, topic, st.session_state.source)
    json_data = st.session_state.results.to_json(indent=2).encode("utf-8")

    st.download_button("📄 Download CSV", csv_data, f"{st.session_state.company}_{topic}_{timestamp}.csv", "text/csv")
    st.download_button("📄 Download PDF", pdf_data, f"{st.session_state.company}_{topic}_{timestamp}.pdf", "application/pdf")
    st.download_button("📄 Download JSON", json_data, f"{st.session_state.company}_{topic}_{timestamp}.json", "application/json")
