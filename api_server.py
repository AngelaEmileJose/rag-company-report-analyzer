from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_system import ImprovedRAGSystem, RAGConfig, HealthCheck, PerformanceMetrics
import shutil
import os
from pathlib import Path

app = FastAPI(title="RAG Company Report Analyzer API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize system
config = RAGConfig.from_env() # Defaults + env vars
metrics = PerformanceMetrics()
rag_system = ImprovedRAGSystem(config)
health_checker = HealthCheck(config, metrics)

class ProcessRequest(BaseModel):
    source: str
    company: str = "Unknown Company"

@app.post("/process")
async def process_document(request: ProcessRequest):
    result = await rag_system.process_document(request.source)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)
    return {"status": "success", "chunks": result.chunk_count, "source": result.source, "time": result.processing_time}

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handle PDF file upload and processing"""
    try:
        # Create temp file path
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the saved file
        result = await rag_system.process_document(str(temp_path))
        
        if not result.success:
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=400, detail=result.message)
            
        return {
            "status": "success", 
            "chunks": result.chunk_count, 
            "source": str(temp_path), # Return local path as source
            "time": result.processing_time,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = rag_system.ask_question(request.question)
    return {"question": request.question, "answer": answer}

class GenerateRequest(BaseModel):
    company: str
    topic: str
    count: int = 5

@app.post("/generate_questions")
async def generate_questions(request: GenerateRequest):
    questions = rag_system.generate_questions(request.company, request.topic, request.count)
    return {"questions": questions}

@app.post("/summary")
async def generate_summary():
    """Generate executive summary for the loaded document"""
    summary = rag_system.generate_executive_summary()
    return {"summary": summary}


@app.get("/health")
async def health_check():
    return await health_checker.check_system_health()

# Serve index.html for root
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
