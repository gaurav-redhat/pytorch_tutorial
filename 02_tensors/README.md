<p align="center">
  <img src="https://img.shields.io/badge/02-Tensors-4CAF50?style=for-the-badge" alt="Tensors"/>
  <img src="https://img.shields.io/badge/Level-Beginner-green?style=for-the-badge" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-30_min-blue?style=for-the-badge" alt="Time"/>
</p>

<h1 align="center">02. Tensors</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../01_basics/README.md">← Prev</a> •
  <a href="../03_autograd/README.md">Next: Autograd →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/02_tensors/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Open_in_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <img src="overview.png" alt="Overview" width="100%"/>
</p>

---

## 🎯 What You'll Learn

| Topic | Description |
|-------|-------------|
| Creating | zeros, ones, rand, randn, arange |
| Reshaping | view, reshape, squeeze, unsqueeze |
| Indexing | Slicing, fancy indexing, boolean |
| Operations | Math, broadcasting, matmul |

---

## 📐 Creating Tensors

```python
import torch

# From data
x = torch.tensor([[1, 2], [3, 4]])

# Zeros and ones
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)

# Random
rand = torch.rand(3, 3)      # Uniform [0, 1)
randn = torch.randn(3, 3)    # Normal (0, 1)

# Range
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Like another tensor
like = torch.zeros_like(x)
```

---

## 🔄 Reshaping

```python
x = torch.arange(12)

# Reshape
x.view(3, 4)      # [12] → [3, 4]
x.view(2, -1)     # [12] → [2, 6]  (-1 = infer)
x.reshape(4, 3)   # Same as view (usually)

# Add/remove dimensions
x.unsqueeze(0)    # [12] → [1, 12]
x.unsqueeze(1)    # [12] → [12, 1]
x.squeeze()       # Remove size-1 dims

# Transpose
x.view(3, 4).T    # [3, 4] → [4, 3]
```

---

## 🎯 Indexing

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic
x[0]          # First row: [1, 2, 3]
x[0, 1]       # Element: 2
x[:, 0]       # First column: [1, 4, 7]

# Slicing
x[0:2]        # First 2 rows
x[:, 1:]      # All rows, cols 1+

# Boolean
x[x > 5]      # Elements > 5: [6, 7, 8, 9]

# Fancy
idx = torch.tensor([0, 2])
x[idx]        # Rows 0 and 2
```

---

## ➕ Operations

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# Element-wise
a + b         # [5, 7, 9]
a * b         # [4, 10, 18]
a ** 2        # [1, 4, 9]

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B     # [3, 5] or torch.matmul(A, B)

# Aggregation
a.sum()       # 6
a.mean()      # 2
a.max()       # 3
```

---

## 🖥️ GPU

```python
# Move to GPU
if torch.cuda.is_available():
    x = x.to('cuda')
    # or
    x = x.cuda()

# Move back to CPU
x = x.to('cpu')
# or
x = x.cpu()
```

---

## ✅ Checklist

- [ ] Create tensors different ways
- [ ] Reshape with view/reshape
- [ ] Index and slice tensors
- [ ] Perform operations
- [ ] Move tensors to GPU

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/02_tensors/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

