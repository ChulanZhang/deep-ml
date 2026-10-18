import numpy as np

def batched_attention_scores(Q, K):
    """
    Compute scaled dot-product attention scores for batched multi-head input.

    Args:
        Q: numpy array of shape (batch, num_heads, seq_len, head_dim)
        K: numpy array of shape (batch, num_heads, seq_len, head_dim)

    Returns:
        Nested list of shape (batch, num_heads, seq_len, seq_len)
    """
    head_dim = Q.shape[-1]

    scores = Q @ K.transpose(0, 1, 3, 2) / head_dim ** 0.5

    return scores.tolist()
