import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    B, T, d_in = x.shape
    d_out = W_out.shape[0]
    head_dim = d_out // num_heads

    qkv = x @ W_qkv # (batch, num_tokens, 3 * d_out)

    Q, K, V = np.split(qkv, 3, axis=-1) # (batch, num_tokens, d_out)

    # (batch, num_heads, num_tokens, head_dim)
    Q = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Q, (batch, num_heads, num_tokens, head_dim)
    # K.transpose(0, 1, 3, 2), (batch, num_heads, head_dim, num_tokens)
    # scores, (B, H, T, T)
    scores = Q @ K.transpose(0, 1, 3, 2) / head_dim ** 0.5

    mask = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(mask, scores, -np.inf)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # attn_weights, (B, H, T, T)
    # V, (B, H, T, D)
    # context, (B, H, T, D)
    context = attn_weights @ V

    # (B, T, H, D)
    context = context.transpose(0, 2, 1, 3)
    # (B, T, d_out)
    context = context.reshape(B, T, d_out)
    # (B, T, d_out)
    output = context @ W_out

    return output.tolist()



