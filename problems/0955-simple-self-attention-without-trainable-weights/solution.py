import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X = np.asarray(X, dtype=float) # (T, d)
    scores = X @ X.T # # (T, T)

    scores = scores - np.max(scores, axis=1, keepdims=True) # (T, T)

    exp_scores = np.exp(scores) # (T, T)

    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (T, T)

    Z = weights @ X # (T, T)

    return Z.tolist()
