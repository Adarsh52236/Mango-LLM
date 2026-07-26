"""
app.py — Gradio ChatGPT-style web interface for the Mango-LLM story generator.

Features:
- Uses gr.ChatInterface for a conversational chat-bubble layout.
- Logs chat turns (user messages and assistant responses) to a Supabase database.
- Uses a unique session_id per application run to group messages into distinct conversations.
- Currently uses a placeholder story generation function that will be swapped for
  actual autoregressive model inference (model.generate) once training is complete.
"""

import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from supabase import create_client, Client

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
    print("Warning: SUPABASE_URL and/or SUPABASE_KEY not found in .env. Chat history will not be saved to database.")

# ---------------------------------------------------------------------------
# 2. Generate Session ID
# ---------------------------------------------------------------------------
# Why generate a session_id?
# --------------------------
# Every time the application starts (or a new app session is launched), generating a unique
# session_id allows us to group all messages exchanged during this specific run into a
# distinct conversation thread in the database. This prevents messages from different users
# or different sessions from getting intermingled in the chat_history table, enabling accurate
# conversation tracking and analytics.

session_id = str(uuid.uuid4())
print(f"Started new chat session with ID: {session_id}")


# ---------------------------------------------------------------------------
# 3. Model Inference (Placeholder)
# ---------------------------------------------------------------------------


def generate_story(prompt: str, max_length: int = 200) -> str:
    """Placeholder function for story generation.

    Will be replaced with actual model loading, tokenizer decoding, and
    model.generate() inference once training finishes.
    """
    return f"[PLACEHOLDER] This will be replaced with real model output. Your prompt was: {prompt}"


# ---------------------------------------------------------------------------
# 4. Chat Interface Handler
# ---------------------------------------------------------------------------


def respond(message: str, history: list) -> str:
    """Handle a chat turn: generate a response and log to Supabase.

    Parameters
    ----------
    message : str
        The latest user message submitted in the chat interface.
    history : list
        The conversation history so far (managed automatically by ChatInterface).

    Returns
    -------
    response : str
        The generated response text to display in the assistant chat bubble.
    """
    # 1. Get response from model
    response = generate_story(message)

    # 2. Insert two rows into chat_history table tagged with the current session_id
    if supabase_client:
        try:
            result = supabase_client.table("chat_history").insert([
                {"session_id": session_id, "role": "user", "message": message},
                {"session_id": session_id, "role": "assistant", "message": response},
            ]).execute()
            if hasattr(result, "error") and result.error:
                print(f"Supabase API Error: {result.error}")
        except Exception as e:
            print(f"Error logging chat turn to Supabase: {e}")

    # 3. Return response text (ChatInterface automatically appends to UI history)
    return response


# ---------------------------------------------------------------------------
# 5. Build and Launch Gradio ChatInterface
# ---------------------------------------------------------------------------
# Why use ChatInterface instead of Interface?
# -------------------------------------------
# The standard gr.Interface is designed for single-shot inputs and outputs (like a single
# textbox in, textbox out), where each interaction is stateless and wipes the previous output.
#
# gr.ChatInterface provides a familiar, modern ChatGPT-style chat-bubble layout that
# automatically manages and renders the back-and-forth conversation history (user bubbles vs.
# assistant bubbles). This creates an interactive, conversational user experience that feels
# like a true AI assistant without requiring manual state management for UI rendering.

demo = gr.ChatInterface(
    fn=respond,
    title="Mango-LLM Chat",
    description="An interactive ChatGPT-style interface for Mango-LLM, a custom causal language model built and trained entirely from scratch (not a fine-tuned existing model).",
)

if __name__ == "__main__":
    demo.launch(share=True)
