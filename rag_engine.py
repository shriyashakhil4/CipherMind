import os
import glob
import uuid
import re
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load the secure .env file
load_dotenv()

# --- 1. SECURE API KEY POOL ---
API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3")
]
API_KEYS = [key for key in API_KEYS if key]
if not API_KEYS:
    print("[FATAL ERROR] No API keys found! Check your .env file.")

# --- 2. MULTI-MODEL FALLBACK ARRAY ---
MODELS = [
    "llama-3.1-8b-instant",         
    "openai/gpt-oss-20b",           
    "groq/compound-mini",           
    "qwen/qwen3.6-27b",             
    "openai/gpt-oss-safeguard-20b", 
    "groq/compound",                
    "llama-3.3-70b-versatile",      
    "openai/gpt-oss-120b"           
]

# Global Tracking Pointers
current_key_idx = 0
current_model_idx = 0

# --- 3. DATABASE & CACHE SETUP ---
chroma_client = chromadb.PersistentClient(path="./chroma_data")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
SESSION_MEMORY = {}

def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 100) -> list:
    """Safely chunks text and forces spaces into massive binary strings to prevent crashes."""
    safe_text = re.sub(r'([^\s]{100})', r'\1 ', text)
    chunks = []
    words = safe_text.split()
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    if words and not chunks:
        chunks.append(" ".join(words))
    return chunks

def rebuild_vector_store(domain: str):
    collection = chroma_client.get_or_create_collection(
        name=f"cipher_mind_{domain}", 
        embedding_function=embedding_fn
    )
    system_dir = os.path.join("knowledge_base", "system_files", domain)
    os.makedirs(system_dir, exist_ok=True)
    
    documents_to_index = []
    metadatas = []
    ids = []
    
    for file_path in glob.glob(os.path.join(system_dir, "*.txt")):
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            chunks = chunk_text(content)
            for index, chunk in enumerate(chunks):
                documents_to_index.append(chunk)
                metadatas.append({"source": file_name, "domain": domain})
                ids.append(f"{domain}_system_{file_name}_chunk_{index}")
        except Exception as e:
            print(f"Skipping unreadable system file {file_name}: {e}")
            
    if documents_to_index:
        collection.upsert(documents=documents_to_index, metadatas=metadatas, ids=ids)

def ephemeral_document_search(extracted_text: str, query: str) -> str:
    """Searches RAM database and limits output chunks to prevent 413 payload errors."""
    try:
        ephemeral_client = chromadb.EphemeralClient()
        collection_name = f"temp_ram_db_{uuid.uuid4().hex}"
        
        temp_collection = ephemeral_client.create_collection(
            name=collection_name, 
            embedding_function=embedding_fn
        )
        
        chunks = chunk_text(extracted_text)
        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            temp_collection.add(
                documents=batch_chunks,
                ids=[f"chunk_{j}" for j in range(i, i + len(batch_chunks))],
                metadatas=[{"source": "temp"} for _ in batch_chunks]
            )
            
        # Pull the top 4 most relevant chunks
        results = temp_collection.query(query_texts=[query], n_results=4)
        context_list = results.get("documents", [[]])[0]
        
        try:
            ephemeral_client.delete_collection(name=collection_name)
        except Exception:
            pass
            
        return "\n\n...".join(context_list) if context_list else ""
    except Exception as e:
        return f"[SYSTEM ERROR: Memory Database Failure - {str(e)}]"

def query_rag(user_message: str, domain: str, session_id: str = None) -> str:
    global current_key_idx, current_model_idx
    
    try:
        try:
            collection = chroma_client.get_collection(
                name=f"cipher_mind_{domain}", 
                embedding_function=embedding_fn
            )
            results = collection.query(query_texts=[user_message], n_results=2)
            context_list = results.get("documents", [[]])[0]
            db_context_str = "\n\n".join(context_list) if context_list else "No relevant background documents found."
        except Exception:
            db_context_str = "Database uninitialized."

        ephemeral_context = ""
        if session_id and session_id in SESSION_MEMORY:
            full_text = SESSION_MEMORY[session_id]
            MAX_DIRECT_CHARS = 4000 
            
            if len(full_text) <= MAX_DIRECT_CHARS:
                # FAST LANE: Small file, inject the whole thing
                ephemeral_context = full_text
            else:
                # HEAVY LANE: Head & Tail Stapler + Vector Search
                targeted_context = ephemeral_document_search(full_text, user_message)
                
                # Snip the top and bottom to guarantee positional context for the AI
                head_snippet = full_text[:1500]
                tail_snippet = full_text[-1500:]
                
                ephemeral_context = (
                    "[SYSTEM NOTE: The user uploaded a large document. "
                    "To ensure you have perfect positional context, here is the START and END of the document, "
                    "followed by specific semantic matches for their query:]\n\n"
                    f"--- START OF DOCUMENT ---\n{head_snippet}...\n\n"
                    f"--- END OF DOCUMENT ---\n...{tail_snippet}\n\n"
                    f"--- TARGETED SEARCH FRAGMENTS ---\n{targeted_context}"
                )

        if domain == "cybersecurity":
            system_role = "You are the CyberMind Threat Matrix, a highly specialized security AI. Analyze the provided context, prioritize technical security measures, and be precise."
        else:
            system_role = "You are the CipherMind Assistant. Answer user prompts professionally and clearly based on the provided context."

        file_injection = f"\n\n--- ACTIVE SESSION DOCUMENT EXTRACT ---\n{ephemeral_context}" if ephemeral_context else ""

        prompt = f"""
        System Instructions: {system_role}
        
        Retrieved Background Knowledge:
        {db_context_str}
        {file_injection}
        
        User Query: {user_message}
        
        Answer the user's query comprehensively using ONLY the provided context.
        """

        # --- 4. THE 24-TIER INFINITE FALLBACK ROUTER ---
        max_attempts = len(API_KEYS) * len(MODELS)
        
        for attempt in range(max_attempts):
            active_key = API_KEYS[current_key_idx]
            active_model = MODELS[current_model_idx]
            
            try:
                temp_client = Groq(api_key=active_key)
                
                chat_completion = temp_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=active_model,
                    temperature=0.3
                )
                
                print(f"[SUCCESS] Routed via Key {current_key_idx + 1} | Model: {active_model}")
                return chat_completion.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(x in error_msg for x in ["429", "413", "rate limit", "token", "tpm", "size"]):
                    print(f"[LIMIT TRIGGERED] Key {current_key_idx + 1} hit restriction on {active_model}. Rotating target...")
                    current_model_idx += 1
                    if current_model_idx >= len(MODELS):
                        current_model_idx = 0
                        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                elif "api_key" in error_msg or "auth" in error_msg:
                    return f"**[AUTHENTICATION ERROR]**: API Key in position {current_key_idx + 1} rejected. Verify your `.env` tokens."
                else:
                    print(f"[RETRYING] Model {active_model} failed with non-quota error: {error_msg}. Advancing loop...")
                    current_model_idx += 1
                    if current_model_idx >= len(MODELS):
                        current_model_idx = 0
                        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    
        return "**[SYSTEM OVERLOAD]**: All 24 free tier token combinations have reached maximum capacity. Please wait 60 seconds for the cloud cooldown cycle to finish."
        
    except Exception as e:
        return f"**[AI ENGINE FAILURE]**: `{str(e)}`"