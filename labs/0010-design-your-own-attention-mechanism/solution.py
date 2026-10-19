import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape
    
    # Your implementation here
    # Hint: Think about how queries "ask questions" and keys "provide answers"
    # 1. Scaled dot-product attention scores
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(dim)

    # 2. Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 3. Weighted sum of values
    output = weights @ V

    return output
