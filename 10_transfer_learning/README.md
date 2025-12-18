<h1 align="center">10. Transfer Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Advanced-F44336?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-45_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../09_transformers/README.md">← Prev: Transformers</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../11_gan/README.md">Next: GANs →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/10_transfer_learning/demo.ipynb">
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
| Pretrained | Load trained models |
| Freeze | Lock backbone weights |
| Fine-tune | Train on your data |

---

## Load Pretrained Model

```python
from torchvision import models

# Load ResNet with ImageNet weights
model = models.resnet50(weights='IMAGENET1K_V1')
```

---

## Freeze Backbone

```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, num_classes)
```

---

## Fine-tune

```python
# Only train the new head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# Or fine-tune everything with lower LR
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

## Checklist

- [ ] Load pretrained model
- [ ] Freeze backbone
- [ ] Replace classification head
- [ ] Fine-tune on your data

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/10_transfer_learning/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../09_transformers/README.md">← Prev: Transformers</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../11_gan/README.md">Next: GANs →</a>
</p>
