<p align="center">
  <img src="https://img.shields.io/badge/Topic-Tensors-4CAF50?style=for-the-badge" alt="Topic"/>
  <img src="https://img.shields.io/badge/Level-Beginner-brightgreen?style=for-the-badge" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-45_min-blue?style=for-the-badge" alt="Time"/>
</p>

<h1 align="center">02. Tensors - The Complete Guide</h1>

<p align="center">
  <a href="../01_basics/README.md">← Basics</a> •
  <a href="../README.md">Home</a> •
  <a href="../03_autograd/README.md">Autograd →</a>
</p>

---

## Overview

![Overview](overview.png)

Tensors are the fundamental data structure in PyTorch. Think of them as multi-dimensional arrays on steroids - they can run on GPUs and track gradients automatically. This tutorial covers **everything** you need to know.

---

## What You'll Learn

- [ ] All ways to create tensors
- [ ] NumPy interoperability
- [ ] Complete tensor operations
- [ ] Linear algebra operations
- [ ] Indexing and slicing
- [ ] Broadcasting rules
- [ ] Memory and performance

---

## 1. Tensor Creation - Every Method

### From Data

```python
import torch
import numpy as np

# From Python list
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]])

# With specific dtype
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
x = torch.tensor([1, 2], dtype=torch.int64)
x = torch.tensor([True, False], dtype=torch.bool)
```

### Initialization Functions

```python
# Zeros and Ones
zeros = torch.zeros(3, 4)           # 3x4 of zeros
ones = torch.ones(2, 3, 4)          # 2x3x4 of ones
full = torch.full((2, 3), 7.0)      # 2x3 filled with 7.0

# Like existing tensor
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)    # Same shape, zeros
ones_like = torch.ones_like(x)      # Same shape, ones

# Empty (uninitialized - faster but random values!)
empty = torch.empty(3, 4)
```

### Random Tensors

```python
# Uniform [0, 1)
rand = torch.rand(3, 4)

# Normal (mean=0, std=1)
randn = torch.randn(3, 4)

# Random integers
randint = torch.randint(low=0, high=10, size=(3, 4))

# Random permutation
perm = torch.randperm(10)  # [7, 2, 4, 0, 1, 9, 5, 6, 8, 3]

# Set seed for reproducibility
torch.manual_seed(42)
```

### Sequences

```python
# Range (like Python range)
x = torch.arange(0, 10)        # [0, 1, 2, ..., 9]
x = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
x = torch.arange(0, 1, 0.1)    # [0.0, 0.1, ..., 0.9]

# Linspace (linearly spaced)
x = torch.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]
x = torch.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]

# Logspace
x = torch.logspace(0, 2, 3)    # [1, 10, 100] = 10^[0,1,2]
```

### Special Matrices

```python
# Identity matrix
eye = torch.eye(3)  # 3x3 identity

# Diagonal matrix
diag = torch.diag(torch.tensor([1, 2, 3]))

# Triangular
lower = torch.tril(torch.ones(3, 3))  # Lower triangular
upper = torch.triu(torch.ones(3, 3))  # Upper triangular
```

---

## 2. NumPy Bridge - Full Interoperability

PyTorch and NumPy share memory when possible. Changes in one affect the other!

### Tensor ↔ NumPy

```python
import numpy as np
import torch

# NumPy to PyTorch
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)    # Shares memory!
tensor = torch.tensor(np_array)        # Copies data

# PyTorch to NumPy (CPU only!)
tensor = torch.tensor([1, 2, 3])
np_array = tensor.numpy()              # Shares memory!
np_array = tensor.detach().numpy()     # Safe way

# GPU tensor to NumPy
gpu_tensor = torch.tensor([1, 2, 3]).cuda()
np_array = gpu_tensor.cpu().numpy()    # Must move to CPU first
```

### Shared Memory Example

```python
# Changes propagate!
a = np.array([1, 2, 3])
b = torch.from_numpy(a)

a[0] = 100
print(b)  # tensor([100, 2, 3]) - Changed!

b[1] = 200
print(a)  # [100, 200, 3] - Also changed!
```

### NumPy-style Operations Work

```python
x = torch.randn(3, 4)

# All these work like NumPy
x.mean()
x.std()
x.var()
x.sum()
x.prod()
x.min()
x.max()
x.argmin()
x.argmax()
```

---

## 3. Tensor Operations - Complete Reference

### Arithmetic Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Basic arithmetic
a + b          # tensor([5., 7., 9.])
a - b          # tensor([-3., -3., -3.])
a * b          # Element-wise: tensor([4., 10., 18.])
a / b          # tensor([0.25, 0.4, 0.5])
a // b         # Floor division
a % b          # Modulo
a ** 2         # Power: tensor([1., 4., 9.])

# In-place operations (end with _)
a.add_(1)      # a = a + 1
a.mul_(2)      # a = a * 2
```

### Math Functions

```python
x = torch.tensor([1.0, 4.0, 9.0])

# Basic math
torch.sqrt(x)      # [1., 2., 3.]
torch.exp(x)       # e^x
torch.log(x)       # ln(x)
torch.log10(x)     # log base 10
torch.log2(x)      # log base 2
torch.abs(x)       # Absolute value

# Trigonometry
torch.sin(x)
torch.cos(x)
torch.tan(x)
torch.asin(x)      # arcsin
torch.atan2(y, x)  # atan(y/x)

# Rounding
torch.floor(x)     # Round down
torch.ceil(x)      # Round up
torch.round(x)     # Round to nearest
torch.trunc(x)     # Truncate decimals
torch.clamp(x, 0, 1)  # Clip values
```

### Reduction Operations

```python
x = torch.tensor([[1., 2., 3.], 
                  [4., 5., 6.]])

# Global reductions
x.sum()           # 21.0
x.mean()          # 3.5
x.std()           # Standard deviation
x.var()           # Variance
x.prod()          # Product of all
x.min()           # Minimum value
x.max()           # Maximum value

# Along axis
x.sum(dim=0)      # [5., 7., 9.] - sum each column
x.sum(dim=1)      # [6., 15.] - sum each row
x.mean(dim=1)     # [2., 5.] - mean of each row

# Keep dimensions
x.sum(dim=1, keepdim=True)  # Shape: (2, 1)

# Multiple dims
x.sum(dim=(0, 1))  # Sum all
```

### Comparison Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])

# Element-wise comparison
a > b             # tensor([False, False, True])
a >= b            # tensor([False, True, True])
a < b             # tensor([True, False, False])
a == b            # tensor([False, True, False])
a != b            # tensor([True, False, True])

# Boolean operations
torch.all(a > 0)  # True if all true
torch.any(a > 2)  # True if any true

# Comparison functions
torch.eq(a, b)    # Same as ==
torch.ne(a, b)    # Not equal
torch.gt(a, b)    # Greater than
torch.lt(a, b)    # Less than

# Max/min between tensors
torch.maximum(a, b)  # Element-wise max
torch.minimum(a, b)  # Element-wise min
```

---

## 4. Linear Algebra - Full Suite

### Matrix Multiplication

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Matrix multiplication (3 equivalent ways)
C = A @ B                  # Preferred
C = torch.mm(A, B)         # Only 2D
C = torch.matmul(A, B)     # Broadcasts

# Batch matrix multiplication
A = torch.randn(10, 3, 4)  # Batch of 10 matrices
B = torch.randn(10, 4, 5)
C = torch.bmm(A, B)        # Shape: (10, 3, 5)

# Matrix-vector
A = torch.randn(3, 4)
x = torch.randn(4)
y = A @ x                  # Shape: (3,)
y = torch.mv(A, x)         # Same thing

# Dot product
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
dot = torch.dot(a, b)      # 32.0

# Outer product
outer = torch.outer(a, b)  # Shape: (3, 3)
```

### Matrix Properties

```python
A = torch.randn(3, 3)

# Transpose
A.T                        # Transpose
A.t()                      # Same for 2D
A.transpose(0, 1)          # Explicit dims
A.permute(1, 0)            # Reorder all dims

# Trace (sum of diagonal)
trace = torch.trace(A)

# Determinant
det = torch.linalg.det(A)

# Rank
rank = torch.linalg.matrix_rank(A)

# Norm
norm = torch.linalg.norm(A)          # Frobenius norm
norm = torch.linalg.norm(A, ord=2)   # Spectral norm
norm = torch.linalg.norm(A, ord='fro')  # Frobenius
norm = torch.linalg.norm(A, ord=float('inf'))  # Max row sum
```

### Matrix Decompositions

```python
A = torch.randn(4, 4)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)

# For symmetric matrices (faster)
eigenvalues, eigenvectors = torch.linalg.eigh(A @ A.T)

# Singular Value Decomposition (SVD)
U, S, Vh = torch.linalg.svd(A)
# A = U @ diag(S) @ Vh

# QR decomposition
Q, R = torch.linalg.qr(A)

# Cholesky (for positive definite)
A_pd = A @ A.T  # Make positive definite
L = torch.linalg.cholesky(A_pd)

# LU decomposition
P, L, U = torch.linalg.lu(A)
```

### Solving Linear Systems

```python
A = torch.randn(3, 3)
b = torch.randn(3)

# Solve Ax = b
x = torch.linalg.solve(A, b)

# Multiple right-hand sides
B = torch.randn(3, 5)
X = torch.linalg.solve(A, B)

# Least squares (overdetermined)
A = torch.randn(10, 3)  # More equations than unknowns
b = torch.randn(10)
x = torch.linalg.lstsq(A, b).solution

# Matrix inverse
A_inv = torch.linalg.inv(A)

# Pseudo-inverse (works for any matrix)
A_pinv = torch.linalg.pinv(A)
```

---

## 5. Reshaping & Indexing

### Shape Operations

```python
x = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)

# View (must be contiguous)
y = x.view(6, 4)          # Reshape to (6, 4)
y = x.view(-1, 4)         # Infer first dim: (6, 4)
y = x.view(-1)            # Flatten: (24,)

# Reshape (works always)
y = x.reshape(6, 4)       # Like view but copies if needed

# Squeeze and unsqueeze
x = torch.randn(1, 3, 1, 4)
x.squeeze()               # Remove all 1s: (3, 4)
x.squeeze(0)              # Remove dim 0: (3, 1, 4)
x.squeeze(2)              # Remove dim 2: (1, 3, 4)

y = torch.randn(3, 4)
y.unsqueeze(0)            # Add dim 0: (1, 3, 4)
y.unsqueeze(1)            # Add dim 1: (3, 1, 4)
y.unsqueeze(-1)           # Add at end: (3, 4, 1)

# Flatten
y = x.flatten()           # All into 1D
y = x.flatten(1)          # Flatten from dim 1
y = x.flatten(1, 2)       # Flatten dims 1-2

# Expand (broadcast to larger)
x = torch.tensor([[1], [2], [3]])  # (3, 1)
y = x.expand(3, 4)        # (3, 4) - copies column

# Repeat (actually copies)
y = x.repeat(2, 3)        # (6, 3)
```

### Concatenation & Stacking

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# Concatenate along existing dim
cat = torch.cat([a, b], dim=0)    # (4, 3)
cat = torch.cat([a, b], dim=1)    # (2, 6)

# Stack along new dim
stack = torch.stack([a, b], dim=0)  # (2, 2, 3)
stack = torch.stack([a, b], dim=1)  # (2, 2, 3)
stack = torch.stack([a, b], dim=2)  # (2, 3, 2)

# Split
parts = torch.chunk(a, 3, dim=1)   # 3 parts along dim 1
parts = torch.split(a, 1, dim=1)   # Each size 1
parts = torch.split(a, [1, 2], dim=1)  # Sizes 1 and 2
```

### Indexing - Everything

```python
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# Basic indexing
x[0]           # tensor([1, 2, 3, 4])
x[0, 1]        # tensor(2)
x[0][1]        # Same as above

# Slicing
x[0:2]         # First 2 rows
x[:, 1:3]      # Columns 1-2
x[0:2, 1:3]    # 2x2 submatrix
x[::2]         # Every other row
x[::-1]        # Reverse rows

# Fancy indexing
indices = torch.tensor([0, 2])
x[indices]     # Rows 0 and 2

# Boolean indexing
mask = x > 5
x[mask]        # tensor([6, 7, 8, 9, 10, 11, 12])
x[x > 5] = 0   # Set all >5 to 0

# Advanced indexing
row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([0, 1, 2])
x[row_idx, col_idx]  # Diagonal: tensor([1, 6, 11])

# Where (conditional)
torch.where(x > 5, x, torch.zeros_like(x))

# Gather (select along dim)
x = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 1], [1, 0]])
torch.gather(x, 1, idx)  # tensor([[1, 2], [4, 3]])

# Scatter (inverse of gather)
src = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 1], [1, 0]])
result = torch.zeros(2, 3)
result.scatter_(1, idx, src)
```

---

## 6. Broadcasting Rules

Broadcasting allows operations on different-shaped tensors:

```python
# Rule: Dimensions are compared from right to left
# Compatible if: equal OR one of them is 1

a = torch.randn(3, 4)
b = torch.randn(4)       # Broadcasts to (1, 4) -> (3, 4)
c = a + b                # Shape: (3, 4)

a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b                # (3, 1) + (1, 4) -> (3, 4)

# More examples
a = torch.randn(2, 3, 4)
b = torch.randn(3, 4)    # -> (1, 3, 4) -> (2, 3, 4)
c = a + b                # Shape: (2, 3, 4)

a = torch.randn(2, 3, 4)
b = torch.randn(4)       # -> (1, 1, 4) -> (2, 3, 4)
c = a + b                # Shape: (2, 3, 4)
```

---

## 7. Memory & Performance

### Contiguous Memory

```python
x = torch.randn(3, 4)
y = x.T                   # Transpose is not contiguous!

y.is_contiguous()         # False
y = y.contiguous()        # Make contiguous (copies)

# view() requires contiguous, reshape() doesn't
x.T.view(-1)              # ERROR
x.T.reshape(-1)           # Works (but copies)
```

### Device Management

```python
# Check device
x = torch.randn(3, 4)
print(x.device)           # cpu

# Move to GPU
if torch.cuda.is_available():
    x = x.to('cuda')
    x = x.cuda()          # Same thing
    
# Move back to CPU
x = x.to('cpu')
x = x.cpu()

# Create on specific device
x = torch.randn(3, 4, device='cuda')

# Operations must be on same device!
a = torch.randn(3, device='cpu')
b = torch.randn(3, device='cuda')
# a + b  # ERROR! Move one first
```

### Memory Efficiency

```python
# In-place operations save memory
x = torch.randn(1000, 1000)
x.add_(1)                 # In-place: no new tensor
x = x + 1                 # Creates new tensor

# Avoid unnecessary copies
a = np.array([1, 2, 3])
b = torch.from_numpy(a)   # Shares memory
b = torch.tensor(a)       # Copies (safer but slower)

# Clone for explicit copy
y = x.clone()             # Deep copy
y = x.detach().clone()    # Clone without grad history
```

---

## Quick Reference Table

| Operation | Code | Description |
|-----------|------|-------------|
| Create zeros | `torch.zeros(3, 4)` | 3x4 zeros |
| Create ones | `torch.ones(3, 4)` | 3x4 ones |
| Random normal | `torch.randn(3, 4)` | N(0,1) |
| Range | `torch.arange(0, 10)` | 0 to 9 |
| Identity | `torch.eye(3)` | 3x3 identity |
| NumPy → Tensor | `torch.from_numpy(arr)` | Share memory |
| Tensor → NumPy | `x.numpy()` | Share memory |
| Reshape | `x.view(2, -1)` | New shape |
| Transpose | `x.T` | Swap dims |
| Matrix multiply | `A @ B` | MatMul |
| Element-wise | `a * b` | Hadamard |
| Sum | `x.sum(dim=0)` | Along dim |
| Boolean mask | `x[x > 0]` | Filter |
| Concatenate | `torch.cat([a,b])` | Join |
| Stack | `torch.stack([a,b])` | New dim |

---

## Try It Yourself

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/02_tensors/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <a href="../01_basics/README.md">← Prev: Basics</a> •
  <a href="../README.md">Back to Main</a> •
  <a href="../03_autograd/README.md">Next: Autograd →</a>
</p>
