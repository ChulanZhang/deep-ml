import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B, T, _ = x.shape
        H, D = self.num_heads, self.d_head

        # B, H, T, D
        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # B, H, T, T
        scores = Q @ K.transpose(3, 2) / D ** 0.5

        if mask is not None:
            scores = scores + mask
        
        weight = torch.softmax(scores, dim=-1)

        # B, H, T, D
        context = weight @ V

        # B, T, H, D
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, T, self.d_model)

        output = self.out_proj(context)

        return output
