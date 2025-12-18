<h1 align="center">06. Training Loop</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-40_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../05_data_loading/README.md">← Prev: Data Loading</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../07_cnn/README.md">Next: CNNs →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/06_training_loop/demo.ipynb">
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
| Forward | pred = model(x) |
| Loss | loss = criterion(pred, y) |
| Backward | loss.backward() |
| Optimize | optimizer.step() |

---

## Complete Training Loop

```python
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for x, y in train_loader:
        # Forward
        pred = model(x)
        loss = criterion(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Loss Functions

```python
nn.CrossEntropyLoss()  # Classification
nn.MSELoss()           # Regression
nn.BCELoss()           # Binary classification
```

---

## Optimizers

```python
torch.optim.SGD(params, lr=0.01)
torch.optim.Adam(params, lr=0.001)
torch.optim.AdamW(params, lr=0.001)
```

---

## Checklist

- [ ] Define model, criterion, optimizer
- [ ] Forward pass: model(x)
- [ ] Compute loss
- [ ] zero_grad → backward → step

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/06_training_loop/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../05_data_loading/README.md">← Prev: Data Loading</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../07_cnn/README.md">Next: CNNs →</a>
</p>
