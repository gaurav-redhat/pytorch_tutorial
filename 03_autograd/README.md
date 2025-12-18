<h1 align="center">03. Autograd</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Beginner-4CAF50?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-20_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../02_tensors/README.md">← Prev: Tensors</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../04_neural_networks/README.md">Next: Neural Networks →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/03_autograd/demo.ipynb">
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
| requires_grad | Enable gradient tracking |
| backward() | Compute gradients |
| grad | Access gradients |
| no_grad | Disable for inference |

---

## Basic Autograd

```python
import torch

# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Forward pass
y = x ** 2 + 3 * x + 1

# Backward pass
y.backward()

# Access gradient (dy/dx = 2x + 3 = 7)
print(x.grad)  # tensor([7.])
```

---

## Computation Graph

```
x (leaf)
    ↓
  x ** 2 → + 3*x → + 1 → y (output)
    ↓
backward() computes dy/dx
```

---

## Control Gradient

```python
# Disable gradients (inference)
with torch.no_grad():
    pred = model(x)

# Detach from graph
y = x.detach()

# Zero gradients
x.grad.zero_()
```

---

## Checklist

- [ ] Create tensor with requires_grad=True
- [ ] Call .backward()
- [ ] Access .grad
- [ ] Use torch.no_grad()

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/03_autograd/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../02_tensors/README.md">← Prev: Tensors</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../04_neural_networks/README.md">Next: Neural Networks →</a>
</p>
