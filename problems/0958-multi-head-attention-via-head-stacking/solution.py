import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    x = np.asarray(x)
    B, T, d_in = x.shape
    outputs = []

    mask = np.tril(np.ones((T, T), dtype=bool))

    for W_q, W_k, W_v in heads:
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v

        d_head = Q.shape[-1]
        scores = Q @ K.transpose(0, 2, 1) / d_head ** 0.5

        scores = np.where(mask, scores, -np.inf)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))

        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        output = attn_weights @ V 

        outputs.append(output)

    return np.concatenate(outputs, axis = -1)

