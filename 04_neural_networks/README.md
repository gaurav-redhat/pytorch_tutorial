<h1 align="center">04. Neural Networks</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Beginner-4CAF50?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-30_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../03_autograd/README.md">← Prev: Autograd</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../05_data_loading/README.md">Next: Data Loading →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/04_neural_networks/demo.ipynb">
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
| nn.Module | Base class for models |
| Layers | Linear, Conv2d, BatchNorm |
| Activations | ReLU, Sigmoid, Softmax |
| Forward | Define computation |

---

## Define a Model

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
```

---

## Common Layers

```python
nn.Linear(in, out)      # Fully connected
nn.Conv2d(c_in, c_out, k)  # Convolution
nn.BatchNorm2d(c)       # Batch norm
nn.Dropout(0.5)         # Dropout
```

---

## Activations

```python
nn.ReLU()       # max(0, x)
nn.Sigmoid()    # 1 / (1 + e^-x)
nn.Tanh()       # tanh(x)
nn.Softmax(dim=1)  # Probabilities
```

---

## Checklist

- [ ] Define class extending nn.Module
- [ ] Define layers in __init__
- [ ] Implement forward()
- [ ] Access parameters with model.parameters()

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/04_neural_networks/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../03_autograd/README.md">← Prev: Autograd</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../05_data_loading/README.md">Next: Data Loading →</a>
</p>
