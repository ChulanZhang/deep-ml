import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.

    Args:
        X: input array of shape (B, S, d_in)
        W_q, W_k, W_v: projection matrices of shape (d_in, d_out)
        W_o: output projection matrix of shape (d_out, d_out)
        num_heads: number of attention heads (must divide d_out)
        causal: if True, apply a causal mask so position t only attends to <= t

    Returns:
        Nested list of shape (B, S, d_out) representing the attention output.
    """
    X = np.asarray(X, dtype=float)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    B, T, d_in = X.shape
    d_out = W_q.shape[-1]

    if d_out % num_heads != 0:
        raise ValueError("d_out must be divisible by num_heads")

    H = num_heads
    D = d_out // H

    # 1. QKV projections
    Q = np.einsum("bti,io->bto", X, W_q)
    K = np.einsum("bti,io->bto", X, W_k)
    V = np.einsum("bti,io->bto", X, W_v)

    # 2. Split into heads: (B, H, T, D)
    Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3)

    # 3. Scaled dot-product attention
    scores = np.einsum("bhtd,bhsd->bhts", Q, K)
    scores = scores / np.sqrt(D)

    # 4. Optional causal mask
    if causal:
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)

    # 5. Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(
        exp_scores, axis=-1, keepdims=True
    )

    # 6. Weighted sum of values
    context = np.einsum("bhts,bhsd->bhtd", weights, V)

    # 7. Merge heads
    context = context.transpose(0, 2, 1, 3)
    context = context.reshape(B, T, d_out)

    # 8. Output projection
    output = np.einsum("bti,io->bto", context, W_o)

    return output.tolist()
