<h1 align="center">07. Convolutional Neural Networks</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-45_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../06_training_loop/README.md">← Prev: Training Loop</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../08_rnn_lstm/README.md">Next: RNNs →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/07_cnn/demo.ipynb">
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
| Conv2d | Convolution layer |
| Pooling | MaxPool, AvgPool |
| Architecture | Conv → Pool → FC |

---

## Convolution Layer

```python
nn.Conv2d(
    in_channels=3,   # RGB
    out_channels=64, # Filters
    kernel_size=3,   # 3x3
    stride=1,
    padding=1
)
```

---

## Pooling

```python
nn.MaxPool2d(2)        # Downsample by 2
nn.AvgPool2d(2)        # Average pooling
nn.AdaptiveAvgPool2d(1) # Global avg pool
```

---

## CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)
```

---

## Checklist

- [ ] Understand Conv2d parameters
- [ ] Use pooling to downsample
- [ ] Build Conv → Pool → FC architecture
- [ ] Train on CIFAR-10

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/07_cnn/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../06_training_loop/README.md">← Prev: Training Loop</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../08_rnn_lstm/README.md">Next: RNNs →</a>
</p>
