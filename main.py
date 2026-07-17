import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

for domain in ["cybersecurity", "general"]:
    os.makedirs(f"knowledge_base/system_files/{domain}", exist_ok=True)

os.makedirs("temp_processing", exist_ok=True)

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
        
        temp_save_path = os.path.join("temp_processing", file.filename)
        with open(temp_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        from document_parser import extract_text_from_file
        extracted_text = extract_text_from_file(temp_save_path)
            
        try:
            os.remove(temp_save_path)
        except Exception:
            pass
        
        active_session = session_id if session_id else "default_session"
        SESSION_MEMORY[active_session] = extracted_text
                
        # THE FIX: Force comprehensive answers even when the user writes a short prompt
        user_msg = message.strip()
        if user_msg:
            final_prompt = (
                f"{user_msg}\n\n"
                f"[SYSTEM DIRECTIVE: The user has uploaded '{file.filename}'. "
                "Analyze the document thoroughly based on their query. Provide a highly detailed, comprehensive, and well-formatted response using bullet points or tables if applicable. Do not give bare-minimum answers.]"
            )
        else:
            final_prompt = (
                f"Analyze this uploaded document: '{file.filename}'. "
                "Provide a highly detailed and comprehensive breakdown. If it contains a list or data, extract and explain it thoroughly."
            )
            
        response_text = query_rag(user_message=final_prompt, domain=domain, session_id=active_session)
        
        return {"response": response_text}
        
    except Exception as e:
        return {"response": f"**[PIPELINE EXCEPTION]**: The backend encountered an error while processing `{file.filename}`.\n\n*Technical Details:* `{str(e)}`"}

@app.post("/api/clear-session")
async def clear_session(session_id: str = Form(...)):
    active_session = session_id if session_id else "default_session"
    if active_session in SESSION_MEMORY:
        del SESSION_MEMORY[active_session]
    return {"status": "success"}