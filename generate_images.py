"""Generate tutorial images for PyTorch Zero to Advanced."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_tutorial_image(title, concepts, filename, color='#EE4C2C'):
    """Create a tutorial overview image."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Background
    bg = patches.FancyBboxPatch((0.1, 0.1), 11.8, 5.8, 
                                  boxstyle="round,pad=0.05",
                                  facecolor='#1a1a2e', edgecolor=color, linewidth=3)
    ax.add_patch(bg)
    
    # Title
    ax.text(6, 5.2, title, fontsize=24, fontweight='bold', 
            color='white', ha='center', va='center')
    
    # Concepts as boxes
    n = len(concepts)
    box_width = 10 / n
    for i, concept in enumerate(concepts):
        x = 1 + i * box_width + box_width/2
        
        # Box
        box = patches.FancyBboxPatch((x - box_width/2 + 0.2, 1.5), 
                                      box_width - 0.4, 2.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor='white', 
                                      linewidth=2, alpha=0.9)
        ax.add_patch(box)
        
        # Concept name
        ax.text(x, 3.2, concept['name'], fontsize=11, fontweight='bold',
                color='white', ha='center', va='center')
        
        # Concept description
        ax.text(x, 2.2, concept['desc'], fontsize=9,
                color='white', ha='center', va='center', alpha=0.9)
    
    # PyTorch logo placeholder
    ax.text(6, 0.7, 'PyTorch', fontsize=14, fontweight='bold',
            color=color, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()

# Tutorial definitions
tutorials = {
    '01_basics': {
        'title': '01. PyTorch Basics',
        'color': '#4CAF50',
        'concepts': [
            {'name': 'Installation', 'desc': 'pip install'},
            {'name': 'Import', 'desc': 'import torch'},
            {'name': 'GPU Check', 'desc': 'cuda.is_available'},
            {'name': 'First Tensor', 'desc': 'torch.tensor()'},
        ]
    },
    '02_tensors': {
        'title': '02. Tensors',
        'color': '#4CAF50',
        'concepts': [
            {'name': 'Create', 'desc': 'zeros, ones, rand'},
            {'name': 'Shape', 'desc': 'view, reshape'},
            {'name': 'Index', 'desc': 'slicing, fancy'},
            {'name': 'Operations', 'desc': 'add, mul, matmul'},
        ]
    },
    '03_autograd': {
        'title': '03. Autograd',
        'color': '#4CAF50',
        'concepts': [
            {'name': 'requires_grad', 'desc': 'Track operations'},
            {'name': 'Backward', 'desc': 'Compute gradients'},
            {'name': 'Graph', 'desc': 'Computational graph'},
            {'name': 'no_grad', 'desc': 'Disable tracking'},
        ]
    },
    '04_neural_networks': {
        'title': '04. Neural Networks',
        'color': '#4CAF50',
        'concepts': [
            {'name': 'nn.Module', 'desc': 'Base class'},
            {'name': 'Layers', 'desc': 'Linear, Conv, etc'},
            {'name': 'Forward', 'desc': 'Define computation'},
            {'name': 'Parameters', 'desc': 'Learnable weights'},
        ]
    },
    '05_data_loading': {
        'title': '05. Data Loading',
        'color': '#FF9800',
        'concepts': [
            {'name': 'Dataset', 'desc': '__getitem__'},
            {'name': 'DataLoader', 'desc': 'Batching'},
            {'name': 'Transforms', 'desc': 'Augmentation'},
            {'name': 'Built-in', 'desc': 'MNIST, CIFAR'},
        ]
    },
    '06_training_loop': {
        'title': '06. Training Loop',
        'color': '#FF9800',
        'concepts': [
            {'name': 'Forward', 'desc': 'Compute output'},
            {'name': 'Loss', 'desc': 'Calculate error'},
            {'name': 'Backward', 'desc': 'Get gradients'},
            {'name': 'Optimize', 'desc': 'Update weights'},
        ]
    },
    '07_cnn': {
        'title': '07. CNNs',
        'color': '#FF9800',
        'concepts': [
            {'name': 'Conv2d', 'desc': 'Feature extraction'},
            {'name': 'Pooling', 'desc': 'Downsample'},
            {'name': 'Flatten', 'desc': 'To vector'},
            {'name': 'Classify', 'desc': 'FC layers'},
        ]
    },
    '08_rnn_lstm': {
        'title': '08. RNNs & LSTMs',
        'color': '#FF9800',
        'concepts': [
            {'name': 'RNN', 'desc': 'Sequential data'},
            {'name': 'LSTM', 'desc': 'Long-term memory'},
            {'name': 'GRU', 'desc': 'Simplified LSTM'},
            {'name': 'Embedding', 'desc': 'Text to vectors'},
        ]
    },
    '09_transformers': {
        'title': '09. Transformers',
        'color': '#F44336',
        'concepts': [
            {'name': 'Attention', 'desc': 'QKV mechanism'},
            {'name': 'Multi-Head', 'desc': 'Parallel attention'},
            {'name': 'Position', 'desc': 'Positional encoding'},
            {'name': 'Encoder', 'desc': 'Full architecture'},
        ]
    },
    '10_transfer_learning': {
        'title': '10. Transfer Learning',
        'color': '#F44336',
        'concepts': [
            {'name': 'Pretrained', 'desc': 'Load weights'},
            {'name': 'Freeze', 'desc': 'Fix layers'},
            {'name': 'Fine-tune', 'desc': 'Train head'},
            {'name': 'HuggingFace', 'desc': 'Model hub'},
        ]
    },
    '11_gan': {
        'title': '11. GANs',
        'color': '#F44336',
        'concepts': [
            {'name': 'Generator', 'desc': 'Create fake data'},
            {'name': 'Discriminator', 'desc': 'Detect fake'},
            {'name': 'Adversarial', 'desc': 'Min-max game'},
            {'name': 'Training', 'desc': 'Alternate updates'},
        ]
    },
    '12_deployment': {
        'title': '12. Deployment',
        'color': '#F44336',
        'concepts': [
            {'name': 'TorchScript', 'desc': 'torch.jit'},
            {'name': 'ONNX', 'desc': 'Export format'},
            {'name': 'Quantize', 'desc': 'Reduce size'},
            {'name': 'Serve', 'desc': 'API/Mobile'},
        ]
    },
}

# Generate images
for folder, info in tutorials.items():
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/overview.png"
    create_tutorial_image(info['title'], info['concepts'], filename, info['color'])
    print(f"Created {filename}")

# Create banner
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

# Background gradient effect
for i in range(100):
    alpha = 0.3 + 0.7 * (i / 100)
    rect = patches.Rectangle((0, i * 0.04), 14, 0.05, 
                               facecolor='#EE4C2C', alpha=alpha * 0.3)
    ax.add_patch(rect)

bg = patches.FancyBboxPatch((0.1, 0.1), 13.8, 3.8,
                              boxstyle="round,pad=0.05",
                              facecolor='#1a1a2e', edgecolor='#EE4C2C', 
                              linewidth=4, alpha=0.95)
ax.add_patch(bg)

# Title
ax.text(7, 2.5, 'PyTorch: Zero to Advanced', fontsize=32, fontweight='bold',
        color='white', ha='center', va='center')
ax.text(7, 1.3, 'Complete Deep Learning Tutorial Series', fontsize=16,
        color='#EE4C2C', ha='center', va='center')
ax.text(7, 0.7, '12 Tutorials  |  Colab Ready  |  Beginner to Expert', fontsize=12,
        color='gray', ha='center', va='center')

plt.tight_layout()
plt.savefig('banner.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Created banner.png")

print("\nAll images generated!")

