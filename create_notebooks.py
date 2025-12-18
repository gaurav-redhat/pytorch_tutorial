"""Create Colab notebooks for all PyTorch tutorials."""
import json
import os

def create_notebook(title, sections, filename):
    """Create a Jupyter notebook with given sections."""
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/pytorch_tutorial/blob/main/" + filename + ")\n",
            "\n",
            "---"
        ]
    })
    
    for section in sections:
        if section['type'] == 'markdown':
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": section['content']
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": section['content']
            })
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"
        },
        "cells": cells
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {filename}")

# 01 Basics
create_notebook("01. PyTorch Basics", [
    {"type": "markdown", "content": ["## Setup\n", "First, let's verify PyTorch is installed."]},
    {"type": "code", "content": ["import torch\n", "print(f'PyTorch version: {torch.__version__}')\n", "print(f'CUDA available: {torch.cuda.is_available()}')\n", "if torch.cuda.is_available():\n", "    print(f'GPU: {torch.cuda.get_device_name(0)}')"]},
    {"type": "markdown", "content": ["## Create Your First Tensor"]},
    {"type": "code", "content": ["# Create a tensor from a list\n", "x = torch.tensor([1, 2, 3, 4, 5])\n", "print(f'Tensor: {x}')\n", "print(f'Shape: {x.shape}')\n", "print(f'Data type: {x.dtype}')"]},
    {"type": "code", "content": ["# Create a 2D tensor\n", "matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])\n", "print(f'Matrix:\\n{matrix}')\n", "print(f'Shape: {matrix.shape}')"]},
    {"type": "markdown", "content": ["## Moving to GPU"]},
    {"type": "code", "content": ["device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n", "print(f'Using device: {device}')\n", "\n", "# Move tensor to GPU\n", "x_gpu = x.to(device)\n", "print(f'Tensor on {x_gpu.device}')"]},
], "01_basics/demo.ipynb")

# 02 Tensors
create_notebook("02. Tensors", [
    {"type": "code", "content": ["import torch"]},
    {"type": "markdown", "content": ["## Creating Tensors"]},
    {"type": "code", "content": ["# Different ways to create tensors\n", "zeros = torch.zeros(3, 4)\n", "ones = torch.ones(2, 3)\n", "rand = torch.rand(3, 3)  # Uniform [0, 1)\n", "randn = torch.randn(3, 3)  # Normal (0, 1)\n", "arange = torch.arange(0, 10, 2)\n", "\n", "print(f'Zeros:\\n{zeros}')\n", "print(f'Rand:\\n{rand}')\n", "print(f'Arange: {arange}')"]},
    {"type": "markdown", "content": ["## Reshaping"]},
    {"type": "code", "content": ["x = torch.arange(12)\n", "print(f'Original: {x}')\n", "print(f'Reshaped (3x4):\\n{x.view(3, 4)}')\n", "print(f'Reshaped (2x-1):\\n{x.view(2, -1)}')  # -1 = infer"]},
    {"type": "markdown", "content": ["## Indexing"]},
    {"type": "code", "content": ["x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n", "print(f'Matrix:\\n{x}')\n", "print(f'First row: {x[0]}')\n", "print(f'Element [1,2]: {x[1, 2]}')\n", "print(f'First column: {x[:, 0]}')\n", "print(f'Elements > 5: {x[x > 5]}')"]},
    {"type": "markdown", "content": ["## Operations"]},
    {"type": "code", "content": ["a = torch.tensor([1., 2., 3.])\n", "b = torch.tensor([4., 5., 6.])\n", "\n", "print(f'a + b = {a + b}')\n", "print(f'a * b = {a * b}')\n", "print(f'a @ b = {a @ b}')  # dot product\n", "print(f'sum(a) = {a.sum()}')\n", "print(f'mean(a) = {a.mean()}')"]},
], "02_tensors/demo.ipynb")

# 03 Autograd
create_notebook("03. Autograd", [
    {"type": "code", "content": ["import torch"]},
    {"type": "markdown", "content": ["## Automatic Differentiation\n", "PyTorch can automatically compute gradients."]},
    {"type": "code", "content": ["# Create tensor with gradient tracking\n", "x = torch.tensor([2.0], requires_grad=True)\n", "\n", "# Forward pass\n", "y = x ** 2  # y = x^2\n", "z = 2 * y + 3  # z = 2x^2 + 3\n", "\n", "# Backward pass\n", "z.backward()\n", "\n", "# Check gradient: dz/dx = 4x = 8\n", "print(f'x = {x.item()}')\n", "print(f'z = {z.item()}')\n", "print(f'dz/dx = {x.grad.item()}')  # Should be 8"]},
    {"type": "markdown", "content": ["## Linear Regression Example"]},
    {"type": "code", "content": ["# Parameters\n", "w = torch.tensor([1.0], requires_grad=True)\n", "b = torch.tensor([0.0], requires_grad=True)\n", "\n", "# Data\n", "x = torch.tensor([1.0, 2.0, 3.0, 4.0])\n", "y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])  # y = 2x\n", "\n", "# Training loop\n", "lr = 0.1\n", "for epoch in range(100):\n", "    # Forward\n", "    y_pred = w * x + b\n", "    loss = ((y_pred - y_true) ** 2).mean()\n", "    \n", "    # Backward\n", "    loss.backward()\n", "    \n", "    # Update\n", "    with torch.no_grad():\n", "        w -= lr * w.grad\n", "        b -= lr * b.grad\n", "        w.grad.zero_()\n", "        b.grad.zero_()\n", "\n", "print(f'Learned: y = {w.item():.2f}x + {b.item():.2f}')\n", "print(f'Expected: y = 2x + 0')"]},
], "03_autograd/demo.ipynb")

# 04 Neural Networks
create_notebook("04. Neural Networks", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn"]},
    {"type": "markdown", "content": ["## Your First Neural Network"]},
    {"type": "code", "content": ["class SimpleNet(nn.Module):\n", "    def __init__(self):\n", "        super().__init__()\n", "        self.fc1 = nn.Linear(784, 128)\n", "        self.fc2 = nn.Linear(128, 64)\n", "        self.fc3 = nn.Linear(64, 10)\n", "        self.relu = nn.ReLU()\n", "    \n", "    def forward(self, x):\n", "        x = self.relu(self.fc1(x))\n", "        x = self.relu(self.fc2(x))\n", "        x = self.fc3(x)\n", "        return x\n", "\n", "model = SimpleNet()\n", "print(model)"]},
    {"type": "code", "content": ["# Count parameters\n", "total_params = sum(p.numel() for p in model.parameters())\n", "print(f'Total parameters: {total_params:,}')"]},
    {"type": "markdown", "content": ["## Using nn.Sequential"]},
    {"type": "code", "content": ["model_seq = nn.Sequential(\n", "    nn.Linear(784, 256),\n", "    nn.ReLU(),\n", "    nn.Dropout(0.2),\n", "    nn.Linear(256, 128),\n", "    nn.ReLU(),\n", "    nn.Linear(128, 10)\n", ")\n", "\n", "# Test forward pass\n", "x = torch.randn(32, 784)  # Batch of 32\n", "output = model_seq(x)\n", "print(f'Input shape: {x.shape}')\n", "print(f'Output shape: {output.shape}')"]},
], "04_neural_networks/demo.ipynb")

# 05 Data Loading
create_notebook("05. Data Loading", [
    {"type": "code", "content": ["import torch\n", "from torch.utils.data import Dataset, DataLoader\n", "from torchvision import datasets, transforms"]},
    {"type": "markdown", "content": ["## Loading MNIST"]},
    {"type": "code", "content": ["# Define transforms\n", "transform = transforms.Compose([\n", "    transforms.ToTensor(),\n", "    transforms.Normalize((0.1307,), (0.3081,))\n", "])\n", "\n", "# Load dataset\n", "train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)\n", "test_data = datasets.MNIST('./data', train=False, transform=transform)\n", "\n", "print(f'Training samples: {len(train_data)}')\n", "print(f'Test samples: {len(test_data)}')"]},
    {"type": "code", "content": ["# Create DataLoader\n", "train_loader = DataLoader(train_data, batch_size=64, shuffle=True)\n", "test_loader = DataLoader(test_data, batch_size=64, shuffle=False)\n", "\n", "# Get a batch\n", "images, labels = next(iter(train_loader))\n", "print(f'Batch images shape: {images.shape}')\n", "print(f'Batch labels shape: {labels.shape}')"]},
    {"type": "markdown", "content": ["## Visualize"]},
    {"type": "code", "content": ["import matplotlib.pyplot as plt\n", "\n", "fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n", "for i, ax in enumerate(axes.flat):\n", "    ax.imshow(images[i].squeeze(), cmap='gray')\n", "    ax.set_title(f'Label: {labels[i].item()}')\n", "    ax.axis('off')\n", "plt.tight_layout()\n", "plt.show()"]},
], "05_data_loading/demo.ipynb")

# 06 Training Loop
create_notebook("06. Training Loop", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import torch.optim as optim\n", "from torch.utils.data import DataLoader\n", "from torchvision import datasets, transforms"]},
    {"type": "markdown", "content": ["## Setup"]},
    {"type": "code", "content": ["device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n", "print(f'Using device: {device}')\n", "\n", "# Data\n", "transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n", "train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)\n", "test_data = datasets.MNIST('./data', train=False, transform=transform)\n", "train_loader = DataLoader(train_data, batch_size=64, shuffle=True)\n", "test_loader = DataLoader(test_data, batch_size=64)"]},
    {"type": "code", "content": ["# Model\n", "model = nn.Sequential(\n", "    nn.Flatten(),\n", "    nn.Linear(784, 256),\n", "    nn.ReLU(),\n", "    nn.Linear(256, 10)\n", ").to(device)\n", "\n", "criterion = nn.CrossEntropyLoss()\n", "optimizer = optim.Adam(model.parameters(), lr=0.001)"]},
    {"type": "markdown", "content": ["## Training"]},
    {"type": "code", "content": ["def train_epoch(model, loader, criterion, optimizer):\n", "    model.train()\n", "    total_loss = 0\n", "    for images, labels in loader:\n", "        images, labels = images.to(device), labels.to(device)\n", "        \n", "        optimizer.zero_grad()\n", "        outputs = model(images)\n", "        loss = criterion(outputs, labels)\n", "        loss.backward()\n", "        optimizer.step()\n", "        \n", "        total_loss += loss.item()\n", "    return total_loss / len(loader)\n", "\n", "def evaluate(model, loader):\n", "    model.eval()\n", "    correct = 0\n", "    with torch.no_grad():\n", "        for images, labels in loader:\n", "            images, labels = images.to(device), labels.to(device)\n", "            outputs = model(images)\n", "            correct += (outputs.argmax(1) == labels).sum().item()\n", "    return correct / len(loader.dataset)"]},
    {"type": "code", "content": ["# Train!\n", "for epoch in range(5):\n", "    loss = train_epoch(model, train_loader, criterion, optimizer)\n", "    acc = evaluate(model, test_loader)\n", "    print(f'Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}')"]},
], "06_training_loop/demo.ipynb")

# 07 CNN
create_notebook("07. CNNs", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import torch.optim as optim\n", "from torch.utils.data import DataLoader\n", "from torchvision import datasets, transforms"]},
    {"type": "markdown", "content": ["## Build a CNN"]},
    {"type": "code", "content": ["class CNN(nn.Module):\n", "    def __init__(self):\n", "        super().__init__()\n", "        self.conv_layers = nn.Sequential(\n", "            nn.Conv2d(1, 32, 3, padding=1),\n", "            nn.ReLU(),\n", "            nn.MaxPool2d(2),\n", "            nn.Conv2d(32, 64, 3, padding=1),\n", "            nn.ReLU(),\n", "            nn.MaxPool2d(2),\n", "        )\n", "        self.fc_layers = nn.Sequential(\n", "            nn.Flatten(),\n", "            nn.Linear(64 * 7 * 7, 128),\n", "            nn.ReLU(),\n", "            nn.Linear(128, 10)\n", "        )\n", "    \n", "    def forward(self, x):\n", "        x = self.conv_layers(x)\n", "        x = self.fc_layers(x)\n", "        return x\n", "\n", "model = CNN()\n", "print(model)"]},
    {"type": "code", "content": ["device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n", "model = model.to(device)\n", "\n", "# Data\n", "transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n", "train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)\n", "test_data = datasets.MNIST('./data', train=False, transform=transform)\n", "train_loader = DataLoader(train_data, batch_size=64, shuffle=True)\n", "test_loader = DataLoader(test_data, batch_size=64)\n", "\n", "criterion = nn.CrossEntropyLoss()\n", "optimizer = optim.Adam(model.parameters())"]},
    {"type": "code", "content": ["# Train\n", "for epoch in range(3):\n", "    model.train()\n", "    for images, labels in train_loader:\n", "        images, labels = images.to(device), labels.to(device)\n", "        optimizer.zero_grad()\n", "        loss = criterion(model(images), labels)\n", "        loss.backward()\n", "        optimizer.step()\n", "    \n", "    # Evaluate\n", "    model.eval()\n", "    correct = 0\n", "    with torch.no_grad():\n", "        for images, labels in test_loader:\n", "            images, labels = images.to(device), labels.to(device)\n", "            correct += (model(images).argmax(1) == labels).sum().item()\n", "    print(f'Epoch {epoch+1}: Accuracy = {correct/len(test_data):.4f}')"]},
], "07_cnn/demo.ipynb")

# 08 RNN/LSTM
create_notebook("08. RNNs & LSTMs", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import torch.nn.functional as F"]},
    {"type": "markdown", "content": ["## Character-Level Language Model"]},
    {"type": "code", "content": ["# Sample text\n", "text = \"hello world, this is a simple example of character level language model.\"\n", "chars = sorted(set(text))\n", "char_to_idx = {c: i for i, c in enumerate(chars)}\n", "idx_to_char = {i: c for c, i in char_to_idx.items()}\n", "vocab_size = len(chars)\n", "print(f'Vocab size: {vocab_size}')\n", "print(f'Characters: {chars}')"]},
    {"type": "code", "content": ["class CharLSTM(nn.Module):\n", "    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):\n", "        super().__init__()\n", "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n", "        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)\n", "        self.fc = nn.Linear(hidden_dim, vocab_size)\n", "    \n", "    def forward(self, x, hidden=None):\n", "        embed = self.embedding(x)\n", "        output, hidden = self.lstm(embed, hidden)\n", "        logits = self.fc(output)\n", "        return logits, hidden\n", "\n", "model = CharLSTM(vocab_size)\n", "print(model)"]},
    {"type": "code", "content": ["# Prepare data\n", "seq_len = 20\n", "data = torch.tensor([char_to_idx[c] for c in text])\n", "\n", "# Create sequences\n", "X, Y = [], []\n", "for i in range(len(data) - seq_len):\n", "    X.append(data[i:i+seq_len])\n", "    Y.append(data[i+1:i+seq_len+1])\n", "X = torch.stack(X)\n", "Y = torch.stack(Y)\n", "print(f'X shape: {X.shape}, Y shape: {Y.shape}')"]},
    {"type": "code", "content": ["# Train\n", "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)\n", "criterion = nn.CrossEntropyLoss()\n", "\n", "for epoch in range(200):\n", "    optimizer.zero_grad()\n", "    logits, _ = model(X)\n", "    loss = criterion(logits.view(-1, vocab_size), Y.view(-1))\n", "    loss.backward()\n", "    optimizer.step()\n", "    if (epoch+1) % 50 == 0:\n", "        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')"]},
    {"type": "code", "content": ["# Generate text\n", "model.eval()\n", "start = 'hello'\n", "generated = list(start)\n", "hidden = None\n", "\n", "x = torch.tensor([[char_to_idx[c] for c in start]])\n", "with torch.no_grad():\n", "    for _ in range(50):\n", "        logits, hidden = model(x, hidden)\n", "        probs = F.softmax(logits[0, -1], dim=-1)\n", "        next_idx = torch.multinomial(probs, 1).item()\n", "        generated.append(idx_to_char[next_idx])\n", "        x = torch.tensor([[next_idx]])\n", "\n", "print('Generated:', ''.join(generated))"]},
], "08_rnn_lstm/demo.ipynb")

# 09 Transformers
create_notebook("09. Transformers", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import math"]},
    {"type": "markdown", "content": ["## Self-Attention from Scratch"]},
    {"type": "code", "content": ["def scaled_dot_product_attention(Q, K, V):\n", "    d_k = Q.size(-1)\n", "    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n", "    attention = torch.softmax(scores, dim=-1)\n", "    return torch.matmul(attention, V), attention\n", "\n", "# Test\n", "seq_len, d_model = 4, 8\n", "Q = K = V = torch.randn(1, seq_len, d_model)\n", "output, attn = scaled_dot_product_attention(Q, K, V)\n", "print(f'Output shape: {output.shape}')\n", "print(f'Attention weights:\\n{attn[0]}')"]},
    {"type": "markdown", "content": ["## Multi-Head Attention"]},
    {"type": "code", "content": ["class MultiHeadAttention(nn.Module):\n", "    def __init__(self, d_model, num_heads):\n", "        super().__init__()\n", "        self.num_heads = num_heads\n", "        self.d_k = d_model // num_heads\n", "        self.W_q = nn.Linear(d_model, d_model)\n", "        self.W_k = nn.Linear(d_model, d_model)\n", "        self.W_v = nn.Linear(d_model, d_model)\n", "        self.W_o = nn.Linear(d_model, d_model)\n", "    \n", "    def forward(self, x):\n", "        batch_size = x.size(0)\n", "        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n", "        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n", "        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n", "        output, _ = scaled_dot_product_attention(Q, K, V)\n", "        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)\n", "        return self.W_o(output)\n", "\n", "mha = MultiHeadAttention(d_model=64, num_heads=4)\n", "x = torch.randn(2, 10, 64)\n", "print(f'MHA output: {mha(x).shape}')"]},
], "09_transformers/demo.ipynb")

# 10 Transfer Learning
create_notebook("10. Transfer Learning", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import torchvision.models as models"]},
    {"type": "markdown", "content": ["## Load Pretrained ResNet"]},
    {"type": "code", "content": ["# Load pretrained model\n", "model = models.resnet18(weights='IMAGENET1K_V1')\n", "print(f'Original classifier: {model.fc}')"]},
    {"type": "code", "content": ["# Freeze all layers\n", "for param in model.parameters():\n", "    param.requires_grad = False\n", "\n", "# Replace classifier for 10 classes\n", "num_classes = 10\n", "model.fc = nn.Sequential(\n", "    nn.Linear(512, 256),\n", "    nn.ReLU(),\n", "    nn.Dropout(0.3),\n", "    nn.Linear(256, num_classes)\n", ")\n", "\n", "print(f'New classifier: {model.fc}')"]},
    {"type": "code", "content": ["# Count trainable parameters\n", "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n", "total = sum(p.numel() for p in model.parameters())\n", "print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')"]},
], "10_transfer_learning/demo.ipynb")

# 11 GAN
create_notebook("11. GANs", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn\n", "import matplotlib.pyplot as plt"]},
    {"type": "markdown", "content": ["## Generator & Discriminator"]},
    {"type": "code", "content": ["class Generator(nn.Module):\n", "    def __init__(self, latent_dim=100):\n", "        super().__init__()\n", "        self.main = nn.Sequential(\n", "            nn.Linear(latent_dim, 256),\n", "            nn.LeakyReLU(0.2),\n", "            nn.Linear(256, 512),\n", "            nn.LeakyReLU(0.2),\n", "            nn.Linear(512, 784),\n", "            nn.Tanh()\n", "        )\n", "    def forward(self, z):\n", "        return self.main(z).view(-1, 1, 28, 28)\n", "\n", "class Discriminator(nn.Module):\n", "    def __init__(self):\n", "        super().__init__()\n", "        self.main = nn.Sequential(\n", "            nn.Flatten(),\n", "            nn.Linear(784, 512),\n", "            nn.LeakyReLU(0.2),\n", "            nn.Linear(512, 256),\n", "            nn.LeakyReLU(0.2),\n", "            nn.Linear(256, 1),\n", "            nn.Sigmoid()\n", "        )\n", "    def forward(self, x):\n", "        return self.main(x)\n", "\n", "G = Generator()\n", "D = Discriminator()\n", "print('Generator:', sum(p.numel() for p in G.parameters()))\n", "print('Discriminator:', sum(p.numel() for p in D.parameters()))"]},
    {"type": "code", "content": ["# Generate random images\n", "z = torch.randn(16, 100)\n", "fake = G(z)\n", "\n", "fig, axes = plt.subplots(2, 8, figsize=(12, 3))\n", "for i, ax in enumerate(axes.flat):\n", "    ax.imshow(fake[i].squeeze().detach(), cmap='gray')\n", "    ax.axis('off')\n", "plt.suptitle('Random (Untrained) Generator Output')\n", "plt.show()"]},
], "11_gan/demo.ipynb")

# 12 Deployment
create_notebook("12. Deployment", [
    {"type": "code", "content": ["import torch\n", "import torch.nn as nn"]},
    {"type": "markdown", "content": ["## TorchScript"]},
    {"type": "code", "content": ["# Simple model\n", "model = nn.Sequential(\n", "    nn.Linear(10, 32),\n", "    nn.ReLU(),\n", "    nn.Linear(32, 5)\n", ")\n", "model.eval()\n", "\n", "# Trace\n", "example = torch.randn(1, 10)\n", "traced = torch.jit.trace(model, example)\n", "\n", "# Save\n", "traced.save('model_traced.pt')\n", "print('Saved traced model')"]},
    {"type": "code", "content": ["# Load and use\n", "loaded = torch.jit.load('model_traced.pt')\n", "output = loaded(torch.randn(5, 10))\n", "print(f'Output shape: {output.shape}')"]},
    {"type": "markdown", "content": ["## Quantization"]},
    {"type": "code", "content": ["# Dynamic quantization\n", "quantized = torch.quantization.quantize_dynamic(\n", "    model, {nn.Linear}, dtype=torch.qint8\n", ")\n", "\n", "# Compare sizes\n", "import os\n", "torch.save(model.state_dict(), 'model_fp32.pt')\n", "torch.save(quantized.state_dict(), 'model_int8.pt')\n", "print(f'FP32 size: {os.path.getsize(\"model_fp32.pt\")/1024:.1f} KB')\n", "print(f'INT8 size: {os.path.getsize(\"model_int8.pt\")/1024:.1f} KB')"]},
], "12_deployment/demo.ipynb")

print("\n✅ All notebooks created!")

