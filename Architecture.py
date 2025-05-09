import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingLib
#print(inspect.getsource(RotaryEmbeddingLib))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # compute RMS over last dim
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * (self.weight / rms)

class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0, "dim must divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.dropout_p = dropout
        # single projection for QKV:
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # rotary on head-dim
        self.rotary = RotaryEmbeddingLib(self.d_head)

    def forward(self, x, key_padding_mask=None):
        # x: (B, S, D)
        B, S, D = x.size()
        # project once to Q, K, V
        qkv = self.qkv_proj(x)  # (B, S, 3*D)
        qkv = qkv.view(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each is (B, S, heads, d_head)
        # transpose for attention: -> (B, heads, S, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # apply RoPE to queries & keys
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)
        # scaled dot-prod attention
        # output: (B, heads, S, d_head)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p,
            is_causal=True
        )
        # back to (B, S, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn_out)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.dropout(self.w2(F.silu(x1) * x2))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int,
                 attn_dropout: float, ff_dropout: float, layer_drop: float):
        super().__init__()
        self.layer_drop = layer_drop
        self.attn_norm = RMSNorm(dim)
        self.ff_norm   = RMSNorm(dim)
        self.attn      = SelfAttentionWithRoPE(dim, n_heads, attn_dropout)
        self.ff        = FeedForward(dim, ff_dim, ff_dropout)

    def forward(self, x, key_padding_mask=None):
        if self.training and torch.rand(1).item() < self.layer_drop:
            return x
        #pre-norm attention
        a = self.attn_norm(x)
        x = x + self.attn(a, key_padding_mask)
        #pre-norm feed-forward
        f = self.ff_norm(x)
        x = x + self.ff(f)
        return x

class ModernTransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq: int,
                 pad_token_id: int,
                 dim: int = 768,
                 n_heads: int = 12,
                 num_layers: int = 70,
                 ff_dim: int = 3072,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 layer_drop: float = 0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        #no learned positions—RoPE everywhere
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim, attn_dropout, dropout, layer_drop)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        #init
        self.apply(self._init_weights)
        #tie embeddings
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.size()
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = input_ids != self.pad_token_id

        x = self.token_emb(input_ids)
        #PyTorch’s MHA uses key_padding_mask=False==pad; pass that into each block
        for block in self.blocks:
            x = block(x, key_padding_mask=~attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return logits, loss
        return logits
  
