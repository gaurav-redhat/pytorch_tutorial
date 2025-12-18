<h1 align="center">08. RNNs & LSTMs</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-45_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../07_cnn/README.md">← Prev: CNNs</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../09_transformers/README.md">Next: Transformers →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/08_rnn_lstm/demo.ipynb">
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
| RNN | Basic recurrence |
| LSTM | Long short-term memory |
| GRU | Gated recurrent unit |

---

## LSTM Layer

```python
lstm = nn.LSTM(
    input_size=128,    # Input features
    hidden_size=256,   # Hidden state size
    num_layers=2,      # Stacked layers
    batch_first=True,  # (batch, seq, features)
    bidirectional=True # Both directions
)

output, (h_n, c_n) = lstm(x)
```

---

## LSTM Model

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

---

## LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | Fewer |
| Memory | Cell state + hidden | Hidden only |

---

## Checklist

- [ ] Understand sequence modeling
- [ ] Use nn.LSTM with batch_first=True
- [ ] Process output and hidden states
- [ ] Try bidirectional

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/08_rnn_lstm/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../07_cnn/README.md">← Prev: CNNs</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../09_transformers/README.md">Next: Transformers →</a>
</p>
