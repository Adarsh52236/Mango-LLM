"""
app.py — Gradio web interface for the Mango-LLM story generator.

Currently uses a placeholder story generation function that will be swapped for
actual autoregressive model inference (model.generate) once training is complete.
"""

import gradio as gr


def generate_story(prompt: str, max_length: int) -> str:
    """Placeholder function for story generation.

    Will be replaced with actual model loading, tokenizer decoding, and
    model.generate() inference once training finishes.
    """
    return f"[PLACEHOLDER] This will be replaced with real model output. Your prompt was: {prompt}"


demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(
            label="Story prompt",
            placeholder="Once upon a time...",
            lines=3,
        ),
        gr.Slider(
            minimum=50,
            maximum=500,
            value=200,
            step=1,
            label="Max length",
        ),
    ],
    outputs=gr.Textbox(label="Generated story", lines=6),
    title="Mango-LLM: A From-Scratch Story Generator",
    description="An interactive demo for Mango-LLM, a custom causal language model built and trained entirely from scratch (not a fine-tuned existing model).",
)

if __name__ == "__main__":
    demo.launch(share=True)
