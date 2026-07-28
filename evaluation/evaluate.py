"""
evaluate.py — Evaluation script for the trained Mango-LLM model.

Downloads/loads the trained model checkpoint from Hugging Face Hub, loads the validation
portion of tokens.bin via memory-mapping, computes average cross-entropy loss over 200
validation batches, and reports the model's perplexity.
"""

import math
import os
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

from model.model import GPTLanguageModel
from data.data import get_batch, val_data, vocab_size

# ---------------------------------------------------------------------------
# Hyperparameters (must match train_colab.py and generate.py exactly)
# ---------------------------------------------------------------------------
embedding_dim = 1024       # size of token / positional embeddings
num_heads     = 16         # number of parallel attention heads per block
num_layers    = 20         # number of stacked transformer blocks
block_size    = 512        # maximum context length
batch_size    = 4          # number of sequences per batch during evaluation
eval_batches  = 200        # number of validation batches to average over

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Download and load checkpoint from Hugging Face Hub
# ---------------------------------------------------------------------------
print("Checking/loading checkpoint_020000.pt from Hugging Face Hub...")
checkpoint_path = hf_hub_download(
    repo_id="AceLeo/mango-llm",
    filename="checkpoint_020000.pt",
    repo_type="model",
)
print(f"Checkpoint loaded from: {checkpoint_path}")

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

# Set the model to evaluation mode
model.eval()
print("Model loaded and set to eval mode.")

# ---------------------------------------------------------------------------
# 3. Load the BPE tokenizer
# ---------------------------------------------------------------------------
_script_dir     = os.path.dirname(os.path.abspath(__file__))
_tokenizer_path = os.path.join(os.path.dirname(_script_dir), "data", "general_bpe.json")
tokenizer       = Tokenizer.from_file(_tokenizer_path)
print("Loaded BPE tokenizer from data/general_bpe.json.")

# ---------------------------------------------------------------------------
# 4. Evaluate on validation set
# ---------------------------------------------------------------------------
# Why use torch.no_grad()?
# ------------------------
# During evaluation and inference, we are only computing forward passes to measure
# model performance; we do not perform backpropagation or update weights.
# Wrapping the evaluation loop in torch.no_grad() tells PyTorch not to build the
# computational graph or store intermediate layer activations for gradients.
# This drastically reduces memory usage (VRAM/RAM) and speeds up computation.

print(f"\nEvaluating model over {eval_batches} validation batches (batch_size={batch_size}, block_size={block_size})...")
print(f"Validation dataset size (memmap view): {len(val_data):,} tokens")

@torch.no_grad()
def evaluate_model() -> tuple[float, float]:
    """Compute average cross-entropy loss and perplexity on the validation set."""
    model.eval()
    losses = torch.zeros(eval_batches)

    for k in range(eval_batches):
        xb, yb = get_batch("val", block_size=block_size, batch_size=batch_size)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(xb, yb)
        losses[k] = loss.item()

    avg_loss = losses.mean().item()
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

if __name__ == "__main__":
    avg_val_loss, val_perplexity = evaluate_model()

    print("\n" + "=" * 60)
    print("Mango-LLM Validation Evaluation Results")
    print("=" * 60)
    print(f"  Average Validation Loss : {avg_val_loss:.4f}")
    print(f"  Validation Perplexity   : {val_perplexity:.2f}")
    print("=" * 60)

    # What is perplexity?
    # -------------------
    # Perplexity is defined as exp(loss), where loss is the average cross-entropy loss.
    # Intuitively, perplexity represents the "effective branching factor" or average
    # uncertainty of the model when predicting the next word. For example, a perplexity
    # of 15 means that at each step, the model is on average as uncertain as if it were
    # guessing uniformly between 15 equally likely next tokens. Lower perplexity
    # indicates superior predictive accuracy and tighter confidence in token generation.
    print(
        "\nNote on Perplexity:\n"
        f"A perplexity of {val_perplexity:.2f} means that when predicting the next token, the model\n"
        f"is on average as uncertain as if it were choosing uniformly among ~{val_perplexity:.0f} possible tokens\n"
        f"out of its total vocabulary of {vocab_size:,}. Lower values indicate better performance."
    )
