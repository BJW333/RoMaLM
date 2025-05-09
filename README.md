# RoMaLM

RoMaLM (Rotary + RMSNorm Language Model) is a modern decoder-only transformer built from scratch in Python.  
It uses Rotary Position Embeddings (RoPE) and RMSNorm for stability and efficiency. 
This is a minimal yet powerful architecture inspired by today’s LLMs and the current trends.

---

## Features

- Rotary Position Embeddings (RoPE)
- RMSNorm (no LayerNorm)
- PreNorm residual connections
- SwiGLU-style feedforward: `SiLU(x1) * x2`
- Tied token embeddings and output head
- Optional LayerDrop for regularization
- Causal attention via `scaled_dot_product_attention`

---

## Model Overview

```python
model = ModernTransformerLM(
    vocab_size=vocab_size,
    max_seq=max_seq_len,
    pad_token_id=pad_token_id,
    dim=768,
    n_heads=12,
    num_layers=8,
    ff_dim=3072,
    dropout=0.1,
    attn_dropout=0.1,
    layer_drop=0.0
)
```
This is just an example. Adjust hyperparameters as needed for your training setup.

---

## License

MIT License — do whatever just don’t claim you wrote it if you didn’t.
