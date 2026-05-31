from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
from chatbot import main
import uuid
import json
import base64

app = FastAPI(title="Logos Christianity AI")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],  # null covers file:// origin
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message:str
    thread_id:Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    thread_id: str = ""
    scripture_ref: Optional[str] = None      # Added
    scripture_text: Optional[str] = None     # Added
    image_url: Optional[str] = None          # Added
    grounded: bool = False                   # Added
    hallucination_warning: bool = False      # Added

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        result = main(req.message, thread_id=thread_id)
    except Exception as e:
        print(f"Error in main(): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
    # Extract last AI message
    ai_msgs = [m for m in result["messages"] if hasattr(m, "content") and m.type == "ai"]
    last_ai = ai_msgs[-1] if ai_msgs else None
    if last_ai is None:
        response_text = "I'm sorry, I could not generate a response."
    elif isinstance(last_ai.content, list):
        # content is a list of blocks e.g. [{"type": "text", "text": "..."}]
        response_text = " ".join(
            block["text"] for block in last_ai.content
            if isinstance(block, dict) and block.get("type") == "text"
        ) or "I'm sorry, I could not generate a response."
    else:
        response_text = last_ai.content or "I'm sorry, I could not generate a response."
 
    # Parse tool results for structured response
    tool_msgs = [m for m in result["messages"] if hasattr(m, "type") and m.type == "tool"]
 
    scripture_ref = scripture_text = image_url = None
    grounded = False
    halluc_warning = False
 
    for tm in tool_msgs:
        try:
            data = json.loads(tm.content) if isinstance(tm.content, str) else tm.content
            if isinstance(data, dict):
                if "context" in data and data.get("found"):
                    grounded = True
                    ctx = data["context"]
                    metas = data.get("metadata", [])
                    if ctx:
                        scripture_text = ctx[0][:300]
                        scripture_ref  = metas[0].get("ref", "") if metas else ""
                if "path" in data and data.get("status") == "success":
                    # Serve the saved image as base64 data URL for simplicity
                    try:
                        with open(data["path"], "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        image_url = f"data:image/png;base64,{b64}"
                    except Exception:
                        image_url = None
        except Exception:
            pass
 
    # Hallucination detection heuristic:
    # If the user's message contained a verse reference and RAG returned nothing
    import re
    verse_pattern = re.compile(r'\b[1-3]?\s?[A-Z][a-z]+\s+\d+:\d+\b')
    if verse_pattern.search(req.message) and not grounded:
        halluc_warning = True
 
    return ChatResponse(
        text=response_text,
        thread_id=thread_id,
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
        image_url=image_url,
        grounded=grounded,
        hallucination_warning=halluc_warning,
    )
 
@app.get("/health")
def health():
    return {"status": "ok", "service": "Logos Christianity AI"}

@app.get("/")
def root():
    return FileResponse("index.html")