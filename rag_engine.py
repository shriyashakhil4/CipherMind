import os
import requests
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# ---------------------------------------------------------
# 1. LIGHTWEIGHT HUGGING FACE API EMBEDDER (The RAM Fix)
# ---------------------------------------------------------
class HuggingFaceAPIEmbedder(EmbeddingFunction):
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        
        if not self.api_token:
            print("[WARNING] HF_API_TOKEN not found in environment variables.")

    def __call__(self, input: Documents) -> Embeddings:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        max_retries = 3
        
        # A robust retry loop to bypass temporary DNS cloud glitches
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json={"inputs": input})
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"HF API Error: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"[NETWORK GLITCH] Connection failed. Retrying in 2 seconds... (Attempt {attempt + 1})")
                    time.sleep(2)
                else:
                    raise Exception(f"Critical Network Failure: Could not reach Hugging Face after {max_retries} attempts. Details: {e}")

# Initialize our custom embedder to pass into our databases
hf_embedder = HuggingFaceAPIEmbedder()


# ---------------------------------------------------------
# 2. CHROMADB VECTOR DATABASE SETUP
# ---------------------------------------------------------
# Connect to the local SQLite database created by Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Create or load the collections for our dual-domain matrix, 
# explicitly attaching our custom lightweight API Embedder
collections = {
    "general": chroma_client.get_or_create_collection(
        name="general_matrix", 
        embedding_function=hf_embedder
    ),
    "cybersecurity": chroma_client.get_or_create_collection(
        name="cyber_matrix", 
        embedding_function=hf_embedder
    )
}

# Temporary RAM storage for conversational context
SESSION_MEMORY = {}


# ---------------------------------------------------------
# 3. THE "HEAD & TAIL STAPLER" CHUNKING ALGORITHM
# ---------------------------------------------------------
def head_and_tail_stapler(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Slices massive documents into overlapping chunks. 
    It 'staples' the tail of the previous chunk to the head of the next
    so the AI never loses context mid-sentence.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but step back by the 'overlap' amount to staple context
        start += (chunk_size - overlap)
        
    return chunks


# ---------------------------------------------------------
# 4. DATABASE INGESTION & QUERY FUNCTIONS
# ---------------------------------------------------------
def rebuild_vector_store(domain: str):
    """
    Reads all text files in the domain's knowledge base folder, 
    chunks them, and uploads them to ChromaDB using the free HF API.
    """
    folder_path = f"knowledge_base/system_files/{domain}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    collection = collections[domain]
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                
            chunks = head_and_tail_stapler(content)
            
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            
            # Because we attached hf_embedder to the collection earlier, 
            # calling add() automatically fires the text to Hugging Face!
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully vectorized and stored {filename} in {domain} matrix.")


def query_rag(user_message: str, domain: str, session_id: str) -> str:
    """
    Takes the user's message, converts it to an embedding, searches the database,
    and returns the context-augmented AI response using Groq.
    """
    # 1. Search the Vector Database
    collection = collections[domain]
    
    # ChromaDB will automatically use our HF API embedder to convert the user_message
    results = collection.query(
        query_texts=[user_message],
        n_results=3 # Get the top 3 most relevant chunks
    )
    
    retrieved_context = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        retrieved_context = "\n".join(results['documents'][0])

    # 2. Manage Conversation Memory
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
        
    SESSION_MEMORY[session_id].append({"role": "user", "content": user_message})
    
    # Keep memory from exploding (limit to last 10 interactions)
    if len(SESSION_MEMORY[session_id]) > 10:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-10:]
        
    # 3. Construct the Augmented Prompt
    system_prompt = f"""You are the CipherMind Intelligence Matrix. 
You are currently operating in the {domain.upper()} domain.

Use the following retrieved context to answer the user's query. 
If the context does not contain the answer, rely on your base knowledge, but prioritize the context.

RETRIEVED CONTEXT:
{retrieved_context}
"""

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(SESSION_MEMORY[session_id])

    # 4. Call Groq API with Cascading Fallback Loop
    groq_keys = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")]
    
    for key in groq_keys:
        if not key:
            continue
            
        try:
            groq_response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "llama3-8b-8192", 
                    "messages": api_messages,
                    "temperature": 0.3
                }
            )
            
            if groq_response.status_code == 200:
                ai_reply = groq_response.json()["choices"][0]["message"]["content"]
                SESSION_MEMORY[session_id].append({"role": "assistant", "content": ai_reply})
                return ai_reply
                
        except Exception as e:
            print(f"Groq API Key failed: {e}. Attempting fallback key...")
            
    return "[SYSTEM ERROR] All Groq API fallbacks failed or rate limit exceeded."