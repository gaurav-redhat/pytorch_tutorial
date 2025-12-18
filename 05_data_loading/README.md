<h1 align="center">05. Data Loading</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-25_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../04_neural_networks/README.md">← Prev: Neural Networks</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../06_training_loop/README.md">Next: Training Loop →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/05_data_loading/demo.ipynb">
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
| Dataset | Custom data class |
| DataLoader | Batching, shuffling |
| Transforms | Preprocessing pipeline |

---

## Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

---

## DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for x, y in loader:
    # x: batch of inputs
    # y: batch of labels
    pass
```

---

## Transforms

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip()
])
```

---

## Checklist

- [ ] Create custom Dataset class
- [ ] Use DataLoader for batching
- [ ] Apply transforms

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/05_data_loading/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../04_neural_networks/README.md">← Prev: Neural Networks</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../06_training_loop/README.md">Next: Training Loop →</a>
</p>
