<h1 align="center">11. GANs</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Level-Advanced-F44336?style=flat-square" alt="Level"/>
  <img src="https://img.shields.io/badge/Time-60_min-blue?style=flat-square" alt="Time"/>
</p>

<p align="center">
  <a href="../10_transfer_learning/README.md">← Prev: Transfer Learning</a> &nbsp;•&nbsp;
  <a href="../README.md">Home</a> &nbsp;•&nbsp;
  <a href="../12_deployment/README.md">Next: Deployment →</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/11_gan/demo.ipynb">
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
| Generator | Creates fake images |
| Discriminator | Detects real vs fake |
| Training | Adversarial game |

---

## Generator

```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # ... more layers
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)
```

---

## Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # ... more layers
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)
```

---

## Training Loop

```python
# Train Discriminator
real_loss = criterion(D(real), ones)
fake_loss = criterion(D(G(z)), zeros)
d_loss = real_loss + fake_loss

# Train Generator
g_loss = criterion(D(G(z)), ones)  # Fool D
```

---

## Checklist

- [ ] Build Generator (upsample)
- [ ] Build Discriminator (downsample)
- [ ] Alternate training D and G
- [ ] Generate images from noise

---

<p align="center">
  <a href="https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/11_gan/demo.ipynb">
    <img src="https://img.shields.io/badge/▶_Run_the_Code-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
  </a>
</p>

<p align="center">
  <a href="../10_transfer_learning/README.md">← Prev: Transfer Learning</a> &nbsp;•&nbsp;
  <a href="../README.md">Back to Main</a> &nbsp;•&nbsp;
  <a href="../12_deployment/README.md">Next: Deployment →</a>
</p>
