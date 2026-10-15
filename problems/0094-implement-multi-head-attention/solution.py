import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    # softmax
    scores_max = torch.amax(scores, dim = -1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    attn_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
    return attn_weights @ V 

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    if n_heads <= 0 or d_model%n_heads != 0:
        return ValueError("n_heads must be positive and divide d_model.")
    d_k = d_model // n_heads
    
    Q_split = Q.reshape(seq_len, n_heads, d_k).transpose(0, 1)
    K_split = K.reshape(seq_len, n_heads, d_k).transpose(0, 1)
    V_split = V.reshape(seq_len, n_heads, d_k).transpose(0, 1)

    heads_out = []
    for i in range(n_heads):
        out_i = self_attention(Q_split[i], K_split[i], V_split[i])
        heads_out.append(out_i)
    
    out = torch.cat(heads_out, dim = -1)

    return out


