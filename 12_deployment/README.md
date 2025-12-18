<h1 align="center">12. Deployment</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Advanced-F44336?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-45_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../11_gan/README.md">← Prev: GANs</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/12_deployment/demo.ipynb">
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
| TorchScript | Serialize for C++ |
| ONNX | Universal format |
| Quantization | Smaller, faster |

---

## TorchScript

```python
# Trace (record operations)
traced = torch.jit.trace(model, example_input)
traced.save("model.pt")

# Script (compile Python)
scripted = torch.jit.script(model)
scripted.save("model.pt")

# Load
loaded = torch.jit.load("model.pt")
```

---

## ONNX Export

```python
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}
)

# Run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

---

## Quantization

```python
# Dynamic quantization (easy)
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Static quantization (better)
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)
# Calibrate with data...
torch.quantization.convert(model, inplace=True)
```

---

## Checklist

- [ ] Export with TorchScript
- [ ] Convert to ONNX
- [ ] Apply quantization
- [ ] Measure speedup

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/12_deployment/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../11_gan/README.md">← Prev: GANs</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a>
</p>

---

<p align="center">
  <b>🎉 Congratulations! You've completed the PyTorch Tutorial Series!</b>
</p>
