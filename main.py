import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import PyPDF2 

load_dotenv()
from rag_engine import query_rag, rebuild_vector_store, SESSION_MEMORY

app = FastAPI(title="CipherMind RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    domain: str
    session_id: str = None

# Ensure required directories exist on startup
for domain in ["cybersecurity", "general"]:
    os.makedirs(f"knowledge_base/system_files/{domain}", exist_ok=True)
os.makedirs("temp_processing", exist_ok=True)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/init-db")
async def initialize_database():
    try:
        rebuild_vector_store("cybersecurity")
        rebuild_vector_store("general")
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if request.domain not in ["cybersecurity", "general"]:
            return {"response": "[SYSTEM ERROR: Invalid domain.]"}

        active_session = request.session_id if request.session_id else "default_session"
        response_text = query_rag(request.message, request.domain, active_session)
        return {"response": response_text}
    except Exception as e:
        return {"response": f"**[BACKEND ROUTING ERROR]**: `{str(e)}`"}

@app.post("/api/chat_with_file")
async def chat_with_file_endpoint(
    file: UploadFile = File(...),
    domain: str = Form(...),
    message: str = Form(""),
    session_id: str = Form("")
):
    try:
        if domain not in ["cybersecurity", "general"]:
            return {"response": "[SYSTEM ERROR: Invalid domain.]"}

        active_session = session_id if session_id else "default_session"

        # 1. Save the uploaded file temporarily
        file_path = f"temp_processing/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract text natively (Safeguarded against append crashes)
        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    # FIX: Using += to build a string, not .append()
                    text_content = page.extract_text()
                    if text_content:
                        extracted_text += text_content + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as text_file:
                extracted_text = text_file.read()

        # 3. Save extracted text to the correct domain's knowledge base
        safe_filename = file.filename.replace(".pdf", ".txt")
        kb_path = f"knowledge_base/system_files/{domain}/{safe_filename}"
        with open(kb_path, "w", encoding="utf-8") as kb_file:
            kb_file.write(extracted_text)

        # 4. Trigger Vector DB rebuild to ingest the new file
        rebuild_vector_store(domain)

        # 5. Query the matrix (Provide default prompt if user only attached file)
        final_message = message if message.strip() else f"Analyze the recently uploaded document named {file.filename}."
        response_text = query_rag(final_message, domain, active_session)

        return {"response": response_text}

    except Exception as e:
        return {"response": f"**[PIPELINE EXCEPTION]**: `{str(e)}`"}

@app.post("/api/clear-session")
async def clear_session_endpoint(session_id: str = Form(...)):
    # Safely wipe memory without converting it to a string type
    if session_id in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = [] 
    return {"status": "success"}