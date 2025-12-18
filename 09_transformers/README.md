<h1 align="center">09. Transformers</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Advanced-F44336?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-60_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../08_rnn_lstm/README.md">← Prev: RNNs</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../10_transfer_learning/README.md">Next: Transfer Learning →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/09_transformers/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

---

## Overview

<p align="center">
  <img src="overview.png" alt="Overview" width="100%"/>
</p>

---

## What You'll Learn

| Topic | Description |
|-------|-------------|
| Self-Attention | Query, Key, Value |
| Multi-Head | Parallel attention |
| Positional Encoding | Sequence position |

---

## Self-Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

```python
def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

---

## Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

---

## Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
```

---

## Checklist

- [ ] Understand Q, K, V
- [ ] Implement scaled dot-product attention
- [ ] Multi-head attention
- [ ] Add positional encoding

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/09_transformers/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../08_rnn_lstm/README.md">← Prev: RNNs</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../10_transfer_learning/README.md">Next: Transfer Learning →</a>
</p>
