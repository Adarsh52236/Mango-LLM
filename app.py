"""
app.py — FastAPI web backend for the Mango-LLM story generator.

Features:
- Pure FastAPI backend using Server-Sent Events (SSE) for streaming text.
- Logs generated conversations to a Supabase database.
"""

import os
import uuid
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import uvicorn
import torch

from generate import model, tokenizer, clean_text, device

# ---------------------------------------------------------------------------
# 1. Load environment variables & initialize Supabase
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Successfully connected to Supabase.")
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
else:
    print("Warning: SUPABASE_URL and/or SUPABASE_KEY not found in .env. Chat history will not be saved.")

session_id = str(uuid.uuid4())
print(f"Started new generation session with ID: {session_id}")

app = FastAPI()

# ---------------------------------------------------------------------------
# 2. FastAPI Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_main():
    """Serve the static index.html landing page."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/chat")
def redirect_chat():
    """Redirect old Gradio route to the main landing page."""
    return RedirectResponse(url="/")


def stream_generation(prompt: str, max_length: int = 200):
    """Generator function that yields tokens one by one as SSE events."""
    token_ids = tokenizer.encode(prompt).ids
    if not token_ids:
        token_ids = [0]
    
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    # We will accumulate the tokens manually
    cleaned_text = ""
    with torch.no_grad():
        for _ in range(max_length):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            
            raw_text = tokenizer.decode(idx[0].tolist())
            cleaned_text = clean_text(raw_text)
            
            payload = json.dumps({"text": cleaned_text})
            yield f"data: {payload}\n\n"
            
    # Send done signal
    yield "data: [DONE]\n\n"
    
    # Log to Supabase when done
    if supabase_client:
        try:
            supabase_client.table("chat_history").insert([
                {"session_id": session_id, "role": "user", "message": prompt},
                {"session_id": session_id, "role": "assistant", "message": cleaned_text},
            ]).execute()
        except Exception as e:
            print(f"Error logging chat turn to Supabase: {e}")


@app.get("/api/generate")
def api_generate(prompt: str):
    """SSE endpoint for streaming text generation."""
    return StreamingResponse(stream_generation(prompt), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
