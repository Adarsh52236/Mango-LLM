"""
generate.py — Text generation script for the trained Mango-LLM model.

Downloads the trained model checkpoint (checkpoint_020000.pt) from Hugging Face Hub,
loads the 20-layer Transformer architecture and BPE tokenizer, and runs autoregressive
story generation from text prompts.
"""

import os
import re
import sys
import torch
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# 1. Install (if needed) and import huggingface_hub
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    import subprocess
    print("huggingface_hub package not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

from model import GPTLanguageModel

# ---------------------------------------------------------------------------
# Hyperparameters (must match train_colab.py exactly)
# ---------------------------------------------------------------------------
embedding_dim = 1024       # size of token / positional embeddings
num_heads     = 16         # number of parallel attention heads per block
num_layers    = 20         # number of stacked transformer blocks
block_size    = 512        # maximum context length
vocab_size    = 8000       # vocabulary size of BPE tokenizer

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Download and load checkpoint from Hugging Face Hub
# ---------------------------------------------------------------------------
# hf_hub_download() downloads checkpoint_020000.pt from the Hugging Face Hub
# ("AceLeo/mango-llm", repo_type="model") into a local cache directory.
# On subsequent runs, it automatically reuses the cached file without re-downloading.
# Note: The checkpoint file is ~4.57 GB, so the initial download may take a few minutes.

print("Checking/downloading checkpoint_020000.pt from Hugging Face Hub...")
checkpoint_path = hf_hub_download(
    repo_id="AceLeo/mango-llm",
    filename="checkpoint_020000.pt",
    repo_type="model",
)
print(f"Checkpoint ready at: {checkpoint_path}")

# Instantiate the model architecture
model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
).to(device)

# Load weights from the checkpoint
print("Loading weights into model...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)

# Set the model to evaluation mode.
# Why eval mode?
# --------------
# During training, neural networks often use stochastic regularization layers
# like Dropout or BatchNorm that behave differently (e.g. randomly zeroing out
# neuron activations to prevent overfitting). Calling model.eval() disables these
# training-only behaviors, ensuring deterministic, full-capacity forward passes
# and accurate, stable predictions during inference and text generation.
model.eval()
print("Model loaded and set to eval mode.")

# ---------------------------------------------------------------------------
# 3. Load the BPE tokenizer
# ---------------------------------------------------------------------------
_script_dir     = os.path.dirname(os.path.abspath(__file__))
_tokenizer_path = os.path.join(_script_dir, "tinystories_bpe.json")
tokenizer       = Tokenizer.from_file(_tokenizer_path)
print("Loaded BPE tokenizer from tinystories_bpe.json.")

# ---------------------------------------------------------------------------
# 4. Post-processing and text generation functions
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Fix spacing around punctuation in generated text.

    Why clean text?
    ---------------
    This fixes an artifact from the BPE tokenizer's Whitespace pre-tokenizer,
    which splits punctuation into separate tokens during training. When decode()
    reconstructs the string, it always inserts a space before every token, causing
    unwanted spaces before punctuation (e.g., "Hello , world .").
    """
    # 1. Remove spaces before basic punctuation marks
    for punct in [",", ".", "!", "?", ";", ":"]:
        text = text.replace(" " + punct, punct)

    # 2. Fix apostrophes inside words (e.g., "it ' s" -> "it's", "don ' t" -> "don't")
    text = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", text)
    text = re.sub(r"(\w)\s+’\s+(\w)", r"\1’\2", text)

    # 3. Remove spaces after opening quotes (double and single, ASCII and Unicode)
    text = re.sub(r'(^|\s|[\(\[\{])"\s+(\S)', r'\1"\2', text)
    text = re.sub(r"(^|\s|[\(\[\{])'\s+(\S)", r"\1'\2", text)
    text = re.sub(r"“\s+", "“", text)
    text = re.sub(r"‘\s+", "‘", text)

    # 4. Remove spaces before closing quotes
    text = re.sub(r'(\S)\s+"($|\s|[.,!?;:\]\)\}])', r'\1"\2', text)
    text = re.sub(r"(\S)\s+'($|\s|[.,!?;:\]\)\}])", r"\1'\2", text)
    text = re.sub(r"\s+”", "”", text)
    text = re.sub(r"\s+’", "’", text)

    return text


@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate text autoregressively from a prompt using Mango-LLM.

    Parameters
    ----------
    prompt : str
        The initial text string to start story generation.
    max_new_tokens : int, optional
        The number of new tokens for the model to generate, by default 200.

    Returns
    -------
    str
        The complete generated text (prompt + generated continuation).
    """
    # 1. Encode the prompt string into BPE token IDs
    token_ids = tokenizer.encode(prompt).ids
    if not token_ids:
        token_ids = [0]  # fallback to pad/bos token if prompt is empty

    # 2. Convert token IDs to a PyTorch tensor of shape (1, T) and move to device
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    # 3. Run autoregressive generation through the model
    out_ids = model.generate(idx, max_new_tokens=max_new_tokens)

    # 4. Decode the resulting token ID tensor back into a Python string
    raw_text = tokenizer.decode(out_ids[0].tolist())

    # 5. Clean punctuation spacing artifacts before returning
    return clean_text(raw_text)

# ---------------------------------------------------------------------------
# 5. Run test generations
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prompts = [
        "Once upon a time",
        "The little cat",
        "One day",
    ]

    print("\n" + "=" * 60)
    print("Running story generations with Mango-LLM")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        generated_story = generate_text(prompt, max_new_tokens=200)
        print(generated_story)
        print("-" * 60)
