"""
model.py — Transformer building blocks for a causal (autoregressive) language model.

Head               – A single scaled dot-product attention head with causal masking.
MultiHeadAttention – Runs multiple Heads in parallel, concatenates outputs, and
                     projects back to embedding_dim.
FeedForward        – A two-layer MLP with 4x inner expansion for added capacity.
Block              – One full transformer layer: LayerNorm -> MultiHeadAttention
                     -> residual -> LayerNorm -> FeedForward -> residual.
GPTLanguageModel   – The complete GPT model: embeddings + Block stack + output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single head of causal (masked) self-attention.

    Given an input of shape (batch, seq_len, embedding_dim), this module:
      1. Projects the input into Query, Key, and Value vectors.
      2. Computes scaled dot-product attention scores.
      3. Applies a causal mask so the model can't peek at future tokens.
      4. Returns the weighted combination of Value vectors.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input embeddings.
    head_size : int
        Dimensionality of the Q, K, V projections for this head.
    """

    def __init__(self, embedding_dim: int, head_size: int):
        super().__init__()

        # --- Linear projections (no bias, following common practice) ---
        # Each one maps from embedding_dim → head_size
        self.query = nn.Linear(embedding_dim, head_size, bias=False)  # "What am I looking for?"
        self.key   = nn.Linear(embedding_dim, head_size, bias=False)  # "What do I contain?"
        self.value = nn.Linear(embedding_dim, head_size, bias=False)  # "What do I communicate?"

        # --- Causal mask ---
        # We pre-register a lower-triangular matrix as a buffer (not a learnable
        # parameter).  It will be used to zero-out attention to future positions.
        # We allocate it at a generous max size; it gets sliced to the actual
        # sequence length in forward().
        #
        # For a sequence of length 4 the mask looks like:
        #
        #   [[1, 0, 0, 0],      ← token 0 can only see token 0
        #    [1, 1, 0, 0],      ← token 1 can see tokens 0-1
        #    [1, 1, 1, 0],      ← token 2 can see tokens 0-2
        #    [1, 1, 1, 1]]      ← token 3 can see tokens 0-3
        #
        # Positions with 0 will be filled with -inf before softmax, which
        # drives their attention weight to zero.
        self.register_buffer("mask", torch.tril(torch.ones(1024, 1024)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, T, C)
            B = batch size
            T = sequence length (number of tokens)
            C = embedding_dim

        Returns
        -------
        out : Tensor of shape (B, T, head_size)
            The attention-weighted output for each position.
        """
        B, T, C = x.shape

        # ----- Step 1: Compute Q, K, V projections -----
        # Each result has shape (B, T, head_size)
        q = self.query(x)   # what each position is looking for
        k = self.key(x)     # what each position advertises
        v = self.value(x)   # what each position will send if attended to

        # ----- Step 2: Compute raw attention scores -----
        # q @ k^T  →  (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        #
        # scores[b][i][j] = how much token i wants to attend to token j
        #
        # We scale by 1/sqrt(head_size) to prevent the dot products from
        # growing too large as head_size increases.  Large values would push
        # softmax into regions with tiny gradients, slowing down learning.
        head_size = q.shape[-1]
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # ----- Step 3: Apply the causal mask -----
        # Why causal masking?
        # We're training a *next-token predictor*:  given tokens [0..t], predict
        # token t+1.  If token t could see tokens t+1, t+2, … the model would
        # just copy the answer instead of learning to predict.  So we mask out
        # all positions j > i (the "future") by setting their scores to -inf.
        # After softmax, -inf becomes 0 → those positions contribute nothing.
        #
        # We slice the pre-built lower-triangular mask to the current seq length.
        causal_mask = self.mask[:T, :T]                         # (T, T)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # ----- Step 4: Softmax → attention weights -----
        # Convert scores to probabilities along the last dim (the "key" dim).
        # Each row sums to 1 and tells us how to mix the Value vectors.
        weights = F.softmax(scores, dim=-1)                     # (B, T, T)

        # ----- Step 5: Weighted aggregation of Values -----
        # weights @ v  →  (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        #
        # For each position i, this computes a weighted sum of Value vectors
        # from positions 0..i (future positions have weight 0 thanks to the mask).
        out = weights @ v                                       # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    Instead of one large attention head, we use *several smaller heads* that
    each independently learn different attention patterns (e.g., one head might
    learn to attend to the previous word, another to the subject of the
    sentence, etc.).

    After all heads produce their outputs, we **concatenate** them along the
    last dimension and pass the result through a linear projection that maps
    back to embedding_dim.  This final projection serves two purposes:
      1. It lets the model *mix* information across heads — the raw
         concatenation is just a stack of independent views; the projection
         learns how to combine them.
      2. It restores the tensor to shape (B, T, embedding_dim), so the
         output can be fed directly into the next layer of the network
         (residual connections, feed-forward blocks, etc.).

    Parameters
    ----------
    num_heads : int
        Number of parallel attention heads.
    embedding_dim : int
        Dimensionality of the model's main embedding vectors.
    head_size : int
        Dimensionality of Q, K, V inside each individual head.
    """

    def __init__(self, num_heads: int, embedding_dim: int, head_size: int):
        super().__init__()

        # Create num_heads independent Head instances.
        # nn.ModuleList (not a plain Python list) is required so that PyTorch
        # can discover the sub-module parameters for gradient updates.
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size) for _ in range(num_heads)]
        )

        # Final linear projection:
        #   (num_heads * head_size) -> embedding_dim
        #
        # Why?
        # Each head outputs a tensor of shape (B, T, head_size).  After
        # concatenation we have (B, T, num_heads * head_size).  The projection
        # compresses this back to (B, T, embedding_dim) so it matches the
        # residual stream of the transformer.
        self.projection = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, T, embedding_dim)

        Returns
        -------
        out : Tensor of shape (B, T, embedding_dim)
        """
        # Run every head on the *same* input independently
        head_outputs = [h(x) for h in self.heads]   # list of (B, T, head_size)

        # Concatenate along the last (feature) dimension:
        #   (B, T, head_size) * num_heads  ->  (B, T, num_heads * head_size)
        concatenated = torch.cat(head_outputs, dim=-1)

        # Project back to embedding_dim so the output can flow into the
        # rest of the network unchanged in shape.
        out = self.projection(concatenated)          # (B, T, embedding_dim)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (a small MLP).

    Applied independently to each position in the sequence.  The hidden layer
    is 4x wider than the input/output — this is **standard practice** in
    transformers ("Attention Is All You Need", Vaswani et al. 2017).

    Why the 4x expansion?
    Attention lets tokens *communicate* with each other, but doesn't do much
    heavy computation per-token.  The feed-forward block gives each token a
    larger internal workspace (4 * embedding_dim neurons) to *process* the
    information it just gathered via attention.  The result is then compressed
    back to embedding_dim so it fits the residual stream.

    Architecture:  Linear(emb -> 4*emb) -> ReLU -> Linear(4*emb -> emb)

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the model's embedding vectors.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            # Expand: embedding_dim -> 4 * embedding_dim
            # This gives the network more capacity to learn complex
            # per-token transformations.
            nn.Linear(embedding_dim, 4 * embedding_dim),

            # Non-linearity — without this the two linear layers would
            # collapse into a single linear transformation.
            nn.ReLU(),

            # Compress back: 4 * embedding_dim -> embedding_dim
            # Restores the original dimension so the output can be added
            # back to the residual stream.
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, embedding_dim) -> (B, T, embedding_dim)"""
        return self.net(x)


class Block(nn.Module):
    """A single transformer block (one "layer" of the transformer).

    Combines multi-head self-attention and a feed-forward network, each
    wrapped with LayerNorm and a residual (skip) connection.

    Architecture ("pre-norm" style):
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Why "pre-norm"?
    The original transformer paper applied LayerNorm *after* each sub-layer
    ("post-norm").  Pre-norm (normalizing *before*) has been shown to train
    more stably, especially for deeper models, and is the dominant style in
    modern LLMs (GPT-2, LLaMA, etc.).

    Why residual connections (the "x +" part)?
    Each sub-layer (attention, feed-forward) transforms the input, but the
    "x +" addition lets the *original* input skip directly around the
    sub-layer.  This has two critical benefits:
      1. **Gradient flow** — during backpropagation, gradients can travel
         straight through the skip connection without being diminished by
         the sub-layer's operations.  This makes deep networks trainable.
      2. **Information preservation** — the sub-layer only needs to learn
         the *delta* (what to add/change), not reconstruct the entire
         representation from scratch each time.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the model's embedding vectors.
    num_heads : int
        Number of parallel attention heads.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()

        # Each head gets an equal slice of the embedding dimension
        head_size = embedding_dim // num_heads

        # Sub-layers
        self.attention   = MultiHeadAttention(num_heads, embedding_dim, head_size)
        self.feedforward = FeedForward(embedding_dim)

        # Layer norms — one before each sub-layer (pre-norm style)
        self.norm1 = nn.LayerNorm(embedding_dim)  # before attention
        self.norm2 = nn.LayerNorm(embedding_dim)  # before feed-forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, T, embedding_dim)

        Returns
        -------
        x : Tensor of shape (B, T, embedding_dim)
        """
        # --- Attention sub-layer with residual connection ---
        # 1. Normalize x
        # 2. Pass through multi-head attention
        # 3. Add the result back to the original x (skip connection)
        x = x + self.attention(self.norm1(x))

        # --- Feed-forward sub-layer with residual connection ---
        # Same pattern: normalize, transform, add back
        x = x + self.feedforward(self.norm2(x))

        return x


class GPTLanguageModel(nn.Module):
    """A complete GPT-style causal language model.

    Assembles all the building blocks into an end-to-end model:

        Token IDs
            |  (token embedding)
            v
        Token embeddings  +  Positional embeddings
            |                       |
            +--------> add <--------+
                        |
                  Block x num_layers
                        |
                    LayerNorm
                        |
                    Linear -> logits  (vocab_size scores per position)

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    embedding_dim : int
        Dimensionality of token / positional embeddings and the residual stream.
    num_heads : int
        Number of attention heads inside each Block.
    num_layers : int
        Number of sequential transformer Blocks.
    block_size : int
        Maximum context length (number of tokens the model can see at once).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
    ):
        super().__init__()
        self.block_size = block_size

        # --- Embedding tables ---
        # Token embedding: each of the vocab_size tokens gets its own
        # learned vector of size embedding_dim.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional embedding: each of the block_size positions gets its
        # own learned vector.  This lets the model know *where* in the
        # sequence each token sits (since attention by itself is
        # permutation-invariant — it has no sense of order without this).
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # --- Transformer backbone ---
        # A stack of num_layers transformer Blocks, applied sequentially.
        # nn.Sequential lets us treat them as a single callable.
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, num_heads) for _ in range(num_layers)]
        )

        # --- Output head ---
        # Final LayerNorm before the projection (matches pre-norm convention).
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Linear projection from embedding_dim -> vocab_size.
        # Produces "logits" — raw (un-normalised) scores for every possible
        # next token at each position in the sequence.
        self.output_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """Run the model on a batch of token-ID sequences.

        Parameters
        ----------
        idx : LongTensor of shape (B, T)
            Input token IDs.  T <= block_size.
        targets : LongTensor of shape (B, T), optional
            Ground-truth next-token IDs for computing cross-entropy loss.
            If None, loss is not computed (e.g. during generation).

        Returns
        -------
        logits : Tensor of shape (B, T, vocab_size)
            Raw prediction scores for the next token at each position.
        loss : Tensor (scalar) or None
            Cross-entropy loss if targets were provided, else None.
        """
        B, T = idx.shape

        # Step 1: Token embeddings  — (B, T) -> (B, T, embedding_dim)
        tok_emb = self.token_embedding(idx)

        # Step 2: Positional embeddings
        # Create position indices [0, 1, 2, ..., T-1] and look them up.
        # The result is (T, embedding_dim); broadcasting adds it to every
        # batch element.
        pos_idx = torch.arange(T, device=idx.device)       # (T,)
        pos_emb = self.position_embedding(pos_idx)          # (T, embedding_dim)

        # Step 3: Combine token + position information
        # Each token now knows *what* it is (token emb) and *where* it sits
        # in the sequence (position emb).
        x = tok_emb + pos_emb                               # (B, T, embedding_dim)

        # Step 4: Pass through the stack of transformer blocks
        x = self.blocks(x)                                  # (B, T, embedding_dim)

        # Step 5: Final layer norm
        x = self.final_norm(x)                              # (B, T, embedding_dim)

        # Step 6: Project to vocabulary logits
        logits = self.output_head(x)                        # (B, T, vocab_size)

        # Step 7: Optionally compute loss
        loss = None
        if targets is not None:
            # F.cross_entropy expects (N, C) and (N,) so we flatten:
            #   logits:  (B, T, vocab_size) -> (B*T, vocab_size)
            #   targets: (B, T)             -> (B*T,)
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively generate new tokens.

        Starting from an initial context `idx`, repeatedly:
          1. Crop to the last block_size tokens (the model's max context).
          2. Run the forward pass to get logits for the next token.
          3. Take the logits at the *last* position (that's the prediction
             for what comes next).
          4. Convert logits -> probabilities via softmax.
          5. Sample one token from that distribution.
          6. Append the sampled token to the running sequence.

        Parameters
        ----------
        idx : LongTensor of shape (B, T)
            Initial context token IDs (can be as short as a single token).
        max_new_tokens : int
            How many new tokens to generate.

        Returns
        -------
        idx : LongTensor of shape (B, T + max_new_tokens)
            The original context with the generated tokens appended.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if it's grown beyond the maximum
            idx_cond = idx[:, -self.block_size:]

            # Forward pass (no targets needed during generation)
            logits, _ = self(idx_cond)

            # We only care about the last time step's predictions
            logits = logits[:, -1, :]                       # (B, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)               # (B, vocab_size)

            # Sample the next token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)         # (B, T+1)

        return idx


# ---------------------------------------------------------------------------
# Quick tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH   = 4
    SEQ_LEN = 8

    # --- Test 1: Single Head ---
    print("=== Test 1: Single Head ===")
    EMBEDDING_DIM = 32
    HEAD_SIZE     = 16

    head = Head(embedding_dim=EMBEDDING_DIM, head_size=HEAD_SIZE)
    x = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    out = head(x)

    print(f"Input  shape: {x.shape}   ->  (batch={BATCH}, seq_len={SEQ_LEN}, emb_dim={EMBEDDING_DIM})")
    print(f"Output shape: {out.shape}  ->  (batch={BATCH}, seq_len={SEQ_LEN}, head_size={HEAD_SIZE})")
    assert out.shape == (BATCH, SEQ_LEN, HEAD_SIZE), "Unexpected output shape!"
    print("[OK] Single Head shape check passed.\n")

    # --- Test 2: MultiHeadAttention ---
    print("=== Test 2: MultiHeadAttention ===")
    NUM_HEADS     = 4
    EMBEDDING_DIM = 32
    HEAD_SIZE     = 8   # 4 heads * 8 head_size = 32 = embedding_dim

    mha = MultiHeadAttention(
        num_heads=NUM_HEADS,
        embedding_dim=EMBEDDING_DIM,
        head_size=HEAD_SIZE,
    )
    x = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    out = mha(x)

    print(f"Input  shape: {x.shape}   ->  (batch={BATCH}, seq_len={SEQ_LEN}, emb_dim={EMBEDDING_DIM})")
    print(f"Output shape: {out.shape}  ->  (batch={BATCH}, seq_len={SEQ_LEN}, emb_dim={EMBEDDING_DIM})")
    print(f"  (num_heads={NUM_HEADS} * head_size={HEAD_SIZE} = {NUM_HEADS * HEAD_SIZE}, projected back to {EMBEDDING_DIM})")
    assert out.shape == (BATCH, SEQ_LEN, EMBEDDING_DIM), "Unexpected output shape!"
    print("[OK] MultiHeadAttention shape check passed.\n")

    # --- Test 3: Block (full transformer layer) ---
    print("=== Test 3: Block ===")
    NUM_HEADS     = 4
    EMBEDDING_DIM = 32

    block = Block(embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS)
    x = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    out = block(x)

    print(f"Input  shape: {x.shape}   ->  (batch={BATCH}, seq_len={SEQ_LEN}, emb_dim={EMBEDDING_DIM})")
    print(f"Output shape: {out.shape}  ->  (batch={BATCH}, seq_len={SEQ_LEN}, emb_dim={EMBEDDING_DIM})")
    print(f"  (head_size = {EMBEDDING_DIM} // {NUM_HEADS} = {EMBEDDING_DIM // NUM_HEADS})")
    assert out.shape == (BATCH, SEQ_LEN, EMBEDDING_DIM), "Unexpected output shape!"
    print("[OK] Block shape check passed.\n")

    # --- Test 4: GPTLanguageModel (full model) ---
    print("=== Test 4: GPTLanguageModel ===")
    VOCAB_SIZE    = 65
    EMBEDDING_DIM = 32
    NUM_HEADS     = 4
    NUM_LAYERS    = 4
    BLOCK_SIZE    = 8

    model = GPTLanguageModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
    )

    # Random token-ID input and targets
    idx     = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    logits, loss = model(idx, targets)

    print(f"Input   shape: {idx.shape}     ->  (batch={BATCH}, seq_len={SEQ_LEN})")
    print(f"Logits  shape: {logits.shape}  ->  (batch={BATCH}, seq_len={SEQ_LEN}, vocab_size={VOCAB_SIZE})")
    print(f"Loss value:    {loss.item():.4f}")
    print(f"  (expected ~-ln(1/65) = {-torch.tensor(1/65).log().item():.4f} for random init)")
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), "Unexpected logits shape!"
    print("[OK] GPTLanguageModel shape & loss check passed.")
