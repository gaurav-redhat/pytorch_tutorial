<p align="center">
  <img src="https://img.shields.io/badge/01-PyTorch_Basics-4CAF50?style=for-the-badge" alt="Basics"/>
  <img src="https://img.shields.io/badge/Level-Beginner-green?style=for-the-badge" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-15_min-blue?style=for-the-badge" alt="Time"/>
</p>

<h1 align="center">01. PyTorch Basics</h1>

<p align="center">
  <a href="../README.md">← Back</a> •
  <a href="../02_tensors/README.md">Next: Tensors →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/01_basics/demo.ipynb">
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
| Installation | How to install PyTorch |
| Import | Import PyTorch and check version |
| GPU | Check if CUDA is available |
| First Tensor | Create your first tensor |

---

## 📦 Installation

```bash
# CPU only
pip install torch torchvision

# With CUDA (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔥 Your First PyTorch Code

```python
import torch

# Check version
print(f"PyTorch version: {torch.__version__}")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create your first tensor
x = torch.tensor([1, 2, 3, 4, 5])
print(f"Tensor: {x}")
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
```

---

## 💡 Key Concepts

### What is PyTorch?
- Deep learning framework by Facebook/Meta
- Dynamic computational graphs
- Python-first design
- Strong GPU acceleration

### What is a Tensor?
- Multi-dimensional array (like NumPy)
- Can run on GPU
- Supports automatic differentiation
- The basic building block

---

## ✅ Checklist

- [ ] Install PyTorch
- [ ] Import torch successfully
- [ ] Check CUDA availability
- [ ] Create a tensor

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/01_basics/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open In Colab"/>
  </a>
</p>

---

<p align="center">
  <a href="../README.md">← Back to Main</a> •
  <a href="../02_tensors/README.md">Next: Tensors →</a>
</p>

