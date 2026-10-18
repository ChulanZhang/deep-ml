import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / d_k ** 0.5

    # stable softmax
    scores = scores - torch.amax(scores, dim = -1, keepdim=True)
    scores = torch.exp(scores)
    attn_weights = scores / torch.sum(scores, dim = -1, keepdim=True)
    output = attn_weights @ V
    return output