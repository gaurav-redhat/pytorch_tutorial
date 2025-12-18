"""Generate high-quality tutorial images for PyTorch Zero to Advanced."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow
import numpy as np
import os

# Set high DPI and better font
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def create_tutorial_image(title, subtitle, sections, filename, accent_color='#EE4C2C'):
    """Create a clean, professional tutorial overview image."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Main background
    main_bg = FancyBboxPatch((0.2, 0.2), 11.6, 6.6,
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor='#161b22', edgecolor=accent_color,
                              linewidth=2)
    ax.add_patch(main_bg)
    
    # Title bar
    title_bar = FancyBboxPatch((0.4, 5.8), 11.2, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=accent_color, edgecolor='none')
    ax.add_patch(title_bar)
    
    # Title text
    ax.text(6, 6.25, title, fontsize=18, fontweight='bold',
            color='white', ha='center', va='center')
    
    # Subtitle
    ax.text(6, 5.5, subtitle, fontsize=11, color='#8b949e',
            ha='center', va='center', style='italic')
    
    # Calculate layout
    n_sections = len(sections)
    section_width = 11 / n_sections
    
    for i, section in enumerate(sections):
        x_center = 0.6 + i * section_width + section_width / 2
        
        # Section header box
        header_box = FancyBboxPatch((x_center - section_width/2 + 0.1, 4.6), 
                                     section_width - 0.2, 0.6,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor='#21262d', edgecolor=accent_color,
                                     linewidth=1.5)
        ax.add_patch(header_box)
        
        # Section name
        ax.text(x_center, 4.9, section['name'], fontsize=11, fontweight='bold',
                color=accent_color, ha='center', va='center')
        
        # Items
        items = section.get('items', [])
        item_height = 4 / max(len(items), 1)
        
        for j, item in enumerate(items):
            y = 4.3 - j * item_height - item_height / 2
            
            # Item background
            item_bg = FancyBboxPatch((x_center - section_width/2 + 0.15, y - item_height/2 + 0.1),
                                      section_width - 0.3, item_height - 0.15,
                                      boxstyle="round,pad=0.01,rounding_size=0.08",
                                      facecolor='#0d1117', edgecolor='#30363d',
                                      linewidth=1)
            ax.add_patch(item_bg)
            
            # Item name (top)
            ax.text(x_center, y + item_height/4, item.get('name', ''),
                    fontsize=9, fontweight='bold', color='#e6edf3',
                    ha='center', va='center')
            
            # Item code (middle)
            if 'code' in item:
                ax.text(x_center, y, item['code'],
                        fontsize=8, fontfamily='monospace', color='#7ee787',
                        ha='center', va='center')
            
            # Item description (bottom)
            if 'desc' in item:
                ax.text(x_center, y - item_height/4, item['desc'],
                        fontsize=7, color='#8b949e',
                        ha='center', va='center')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(filename, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Created: {filename}")


# Define all tutorials
tutorials = {
    '01_basics': {
        'title': 'PyTorch Basics',
        'subtitle': 'Installation, First Tensor, GPU Setup',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Install', 'items': [
                {'name': 'pip install', 'code': 'pip install torch', 'desc': 'CPU version'},
                {'name': 'with CUDA', 'code': 'torch+cu118', 'desc': 'GPU version'},
            ]},
            {'name': 'Import', 'items': [
                {'name': 'PyTorch', 'code': 'import torch', 'desc': 'Main module'},
                {'name': 'Version', 'code': 'torch.__version__', 'desc': 'Check install'},
            ]},
            {'name': 'GPU Check', 'items': [
                {'name': 'Available?', 'code': 'cuda.is_available()', 'desc': 'True/False'},
                {'name': 'Device', 'code': "device('cuda')", 'desc': 'Select GPU'},
            ]},
            {'name': 'First Tensor', 'items': [
                {'name': 'Create', 'code': 'tensor([1,2,3])', 'desc': 'From list'},
                {'name': 'Properties', 'code': '.shape .dtype', 'desc': 'Inspect'},
            ]},
        ]
    },
    '02_tensors': {
        'title': 'Tensors - Complete Guide',
        'subtitle': 'Creation, NumPy, Operations, Linear Algebra',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Creation', 'items': [
                {'name': 'Zeros/Ones', 'code': 'zeros(3,4)', 'desc': 'Fill values'},
                {'name': 'Random', 'code': 'randn(3,3)', 'desc': 'Normal dist'},
                {'name': 'Range', 'code': 'arange(0,10)', 'desc': 'Sequence'},
                {'name': 'Identity', 'code': 'eye(3)', 'desc': 'I matrix'},
            ]},
            {'name': 'NumPy', 'items': [
                {'name': 'To NumPy', 'code': '.numpy()', 'desc': 'Share memory'},
                {'name': 'From NumPy', 'code': 'from_numpy()', 'desc': 'Share memory'},
                {'name': 'Copy', 'code': 'tensor(arr)', 'desc': 'New memory'},
            ]},
            {'name': 'Linear Algebra', 'items': [
                {'name': 'MatMul', 'code': 'A @ B', 'desc': 'Matrix multiply'},
                {'name': 'Inverse', 'code': 'linalg.inv()', 'desc': 'A^-1'},
                {'name': 'SVD', 'code': 'linalg.svd()', 'desc': 'Decompose'},
                {'name': 'Solve', 'code': 'linalg.solve()', 'desc': 'Ax = b'},
            ]},
            {'name': 'Reshape', 'items': [
                {'name': 'View', 'code': '.view(2,3)', 'desc': 'New shape'},
                {'name': 'Squeeze', 'code': '.squeeze()', 'desc': 'Remove 1s'},
                {'name': 'Cat', 'code': 'cat([a,b])', 'desc': 'Concatenate'},
                {'name': 'Stack', 'code': 'stack([a,b])', 'desc': 'New dim'},
            ]},
        ]
    },
    '03_autograd': {
        'title': 'Autograd',
        'subtitle': 'Automatic Differentiation Engine',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Enable Gradients', 'items': [
                {'name': 'Track', 'code': 'requires_grad=True', 'desc': 'Enable'},
                {'name': 'Compute', 'code': 'loss.backward()', 'desc': 'Backprop'},
                {'name': 'Access', 'code': 'x.grad', 'desc': 'Get gradient'},
            ]},
            {'name': 'Graph', 'items': [
                {'name': 'Leaf', 'code': 'x (input)', 'desc': 'Start'},
                {'name': 'Operation', 'code': 'y = f(x)', 'desc': 'Transform'},
                {'name': 'grad_fn', 'code': '.grad_fn', 'desc': 'History'},
            ]},
            {'name': 'Control', 'items': [
                {'name': 'Disable', 'code': 'with no_grad():', 'desc': 'Inference'},
                {'name': 'Detach', 'code': '.detach()', 'desc': 'Remove'},
                {'name': 'Zero', 'code': '.zero_()', 'desc': 'Reset'},
            ]},
        ]
    },
    '04_neural_networks': {
        'title': 'Neural Networks',
        'subtitle': 'nn.Module, Layers, Activations',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Layers', 'items': [
                {'name': 'Linear', 'code': 'nn.Linear(in,out)', 'desc': 'Dense'},
                {'name': 'Conv2d', 'code': 'nn.Conv2d(c,c,k)', 'desc': 'Convolution'},
                {'name': 'BatchNorm', 'code': 'nn.BatchNorm2d', 'desc': 'Normalize'},
                {'name': 'Dropout', 'code': 'nn.Dropout(0.5)', 'desc': 'Regularize'},
            ]},
            {'name': 'Activations', 'items': [
                {'name': 'ReLU', 'code': 'nn.ReLU()', 'desc': 'max(0,x)'},
                {'name': 'Sigmoid', 'code': 'nn.Sigmoid()', 'desc': '[0,1]'},
                {'name': 'Softmax', 'code': 'nn.Softmax()', 'desc': 'Probs'},
            ]},
            {'name': 'Model', 'items': [
                {'name': 'Class', 'code': 'class Net(Module)', 'desc': 'Define'},
                {'name': 'Forward', 'code': 'def forward():', 'desc': 'Compute'},
                {'name': 'Params', 'code': '.parameters()', 'desc': 'Weights'},
            ]},
        ]
    },
    '05_data_loading': {
        'title': 'Data Loading',
        'subtitle': 'Dataset, DataLoader, Transforms',
        'color': '#FF9800',
        'sections': [
            {'name': 'Dataset', 'items': [
                {'name': 'Length', 'code': '__len__()', 'desc': 'Size'},
                {'name': 'Get Item', 'code': '__getitem__()', 'desc': 'Sample'},
                {'name': 'Transform', 'code': 'transform(x)', 'desc': 'Process'},
            ]},
            {'name': 'DataLoader', 'items': [
                {'name': 'Batch', 'code': 'batch_size=32', 'desc': 'Group'},
                {'name': 'Shuffle', 'code': 'shuffle=True', 'desc': 'Random'},
                {'name': 'Workers', 'code': 'num_workers=4', 'desc': 'Parallel'},
            ]},
            {'name': 'Transforms', 'items': [
                {'name': 'ToTensor', 'code': 'ToTensor()', 'desc': 'Convert'},
                {'name': 'Normalize', 'code': 'Normalize(m,s)', 'desc': 'Scale'},
                {'name': 'Compose', 'code': 'Compose([...])', 'desc': 'Chain'},
            ]},
        ]
    },
    '06_training_loop': {
        'title': 'Training Loop',
        'subtitle': 'Forward, Backward, Optimize',
        'color': '#FF9800',
        'sections': [
            {'name': 'Forward', 'items': [
                {'name': 'Predict', 'code': 'out = model(x)', 'desc': 'Inference'},
                {'name': 'Loss', 'code': 'loss = criterion()', 'desc': 'Error'},
            ]},
            {'name': 'Backward', 'items': [
                {'name': 'Zero', 'code': 'optimizer.zero_grad()', 'desc': 'Clear'},
                {'name': 'Backprop', 'code': 'loss.backward()', 'desc': 'Gradients'},
                {'name': 'Update', 'code': 'optimizer.step()', 'desc': 'Weights'},
            ]},
            {'name': 'Loss Functions', 'items': [
                {'name': 'CE', 'code': 'CrossEntropyLoss()', 'desc': 'Classify'},
                {'name': 'MSE', 'code': 'MSELoss()', 'desc': 'Regress'},
                {'name': 'BCE', 'code': 'BCELoss()', 'desc': 'Binary'},
            ]},
            {'name': 'Optimizers', 'items': [
                {'name': 'SGD', 'code': 'optim.SGD()', 'desc': 'Classic'},
                {'name': 'Adam', 'code': 'optim.Adam()', 'desc': 'Adaptive'},
                {'name': 'AdamW', 'code': 'optim.AdamW()', 'desc': 'W decay'},
            ]},
        ]
    },
    '07_cnn': {
        'title': 'Convolutional Neural Networks',
        'subtitle': 'Conv2d, Pooling, Image Classification',
        'color': '#FF9800',
        'sections': [
            {'name': 'Convolution', 'items': [
                {'name': 'Conv2d', 'code': 'Conv2d(3,64,3)', 'desc': 'in,out,kernel'},
                {'name': 'Stride', 'code': 'stride=2', 'desc': 'Step size'},
                {'name': 'Padding', 'code': 'padding=1', 'desc': 'Border'},
            ]},
            {'name': 'Pooling', 'items': [
                {'name': 'MaxPool', 'code': 'MaxPool2d(2)', 'desc': 'Downsample'},
                {'name': 'AvgPool', 'code': 'AvgPool2d(2)', 'desc': 'Average'},
                {'name': 'Adaptive', 'code': 'AdaptiveAvgPool', 'desc': 'Fixed out'},
            ]},
            {'name': 'Architecture', 'items': [
                {'name': 'Input', 'code': '[B,C,H,W]', 'desc': 'Image batch'},
                {'name': 'Features', 'code': 'Conv+ReLU+Pool', 'desc': 'Extract'},
                {'name': 'Classifier', 'code': 'Flatten+FC', 'desc': 'Predict'},
            ]},
        ]
    },
    '08_rnn_lstm': {
        'title': 'RNNs & LSTMs',
        'subtitle': 'Sequential Data, Time Series, NLP',
        'color': '#FF9800',
        'sections': [
            {'name': 'RNN', 'items': [
                {'name': 'Input', 'code': 'x_t', 'desc': 'Current'},
                {'name': 'Hidden', 'code': 'h_t', 'desc': 'Memory'},
                {'name': 'Update', 'code': 'h = tanh(Wx+Uh)', 'desc': 'Combine'},
            ]},
            {'name': 'LSTM', 'items': [
                {'name': 'Forget', 'code': 'f = sigmoid()', 'desc': 'What to drop'},
                {'name': 'Input', 'code': 'i = sigmoid()', 'desc': 'What to add'},
                {'name': 'Cell', 'code': 'c = f*c + i*g', 'desc': 'Long memory'},
            ]},
            {'name': 'PyTorch', 'items': [
                {'name': 'LSTM', 'code': 'nn.LSTM(in,h)', 'desc': 'Layer'},
                {'name': 'GRU', 'code': 'nn.GRU(in,h)', 'desc': 'Simpler'},
                {'name': 'BiDir', 'code': 'bidirectional', 'desc': 'Both ways'},
            ]},
        ]
    },
    '09_transformers': {
        'title': 'Transformers',
        'subtitle': 'Self-Attention, Multi-Head, Positional Encoding',
        'color': '#F44336',
        'sections': [
            {'name': 'Attention', 'items': [
                {'name': 'Query', 'code': 'Q = xW_q', 'desc': 'What to find'},
                {'name': 'Key', 'code': 'K = xW_k', 'desc': 'What I have'},
                {'name': 'Value', 'code': 'V = xW_v', 'desc': 'Content'},
                {'name': 'Score', 'code': 'softmax(QK/d)', 'desc': 'Weight'},
            ]},
            {'name': 'Multi-Head', 'items': [
                {'name': 'Heads', 'code': 'num_heads=8', 'desc': 'Parallel'},
                {'name': 'Concat', 'code': 'cat(h1..h8)', 'desc': 'Combine'},
                {'name': 'Project', 'code': 'output @ W_o', 'desc': 'Final'},
            ]},
            {'name': 'Position', 'items': [
                {'name': 'Sinusoidal', 'code': 'sin/cos', 'desc': 'Fixed'},
                {'name': 'Learned', 'code': 'Embedding', 'desc': 'Trainable'},
                {'name': 'RoPE', 'code': 'Rotary', 'desc': 'Modern'},
            ]},
        ]
    },
    '10_transfer_learning': {
        'title': 'Transfer Learning',
        'subtitle': 'Pretrained Models, Fine-tuning',
        'color': '#F44336',
        'sections': [
            {'name': 'Load', 'items': [
                {'name': 'ResNet', 'code': 'resnet50(weights)', 'desc': 'ImageNet'},
                {'name': 'BERT', 'code': 'from_pretrained()', 'desc': 'NLP'},
                {'name': 'ViT', 'code': 'vit_b_16()', 'desc': 'Vision'},
            ]},
            {'name': 'Freeze', 'items': [
                {'name': 'All', 'code': 'requires_grad=False', 'desc': 'Lock'},
                {'name': 'Backbone', 'code': 'model.features', 'desc': 'Keep'},
                {'name': 'Gradual', 'code': 'Unfreeze layers', 'desc': 'Stage'},
            ]},
            {'name': 'Fine-tune', 'items': [
                {'name': 'Replace', 'code': 'model.fc = ...', 'desc': 'New head'},
                {'name': 'Low LR', 'code': 'lr=1e-5', 'desc': 'Careful'},
                {'name': 'Train', 'code': 'Standard loop', 'desc': 'Your data'},
            ]},
        ]
    },
    '11_gan': {
        'title': 'GANs',
        'subtitle': 'Generator, Discriminator, Adversarial Training',
        'color': '#F44336',
        'sections': [
            {'name': 'Generator', 'items': [
                {'name': 'Input', 'code': 'z ~ N(0,1)', 'desc': 'Noise'},
                {'name': 'Upsample', 'code': 'ConvTranspose2d', 'desc': 'Grow'},
                {'name': 'Output', 'code': 'Tanh()', 'desc': 'Fake image'},
            ]},
            {'name': 'Discriminator', 'items': [
                {'name': 'Input', 'code': 'Real/Fake img', 'desc': 'Image'},
                {'name': 'Down', 'code': 'Conv2d stride=2', 'desc': 'Shrink'},
                {'name': 'Output', 'code': 'Sigmoid() 0/1', 'desc': 'Real prob'},
            ]},
            {'name': 'Training', 'items': [
                {'name': 'D Loss', 'code': 'BCE(D(x),1)+...', 'desc': 'Detect'},
                {'name': 'G Loss', 'code': 'BCE(D(G(z)),1)', 'desc': 'Fool D'},
                {'name': 'Alternate', 'code': 'Train D, G', 'desc': 'Each step'},
            ]},
        ]
    },
    '12_deployment': {
        'title': 'Deployment',
        'subtitle': 'TorchScript, ONNX, Quantization',
        'color': '#F44336',
        'sections': [
            {'name': 'TorchScript', 'items': [
                {'name': 'Trace', 'code': 'jit.trace(m,x)', 'desc': 'Record'},
                {'name': 'Script', 'code': 'jit.script(m)', 'desc': 'Compile'},
                {'name': 'Save', 'code': 'model.save()', 'desc': 'Export'},
            ]},
            {'name': 'ONNX', 'items': [
                {'name': 'Export', 'code': 'onnx.export()', 'desc': 'Universal'},
                {'name': 'Runtime', 'code': 'onnxruntime', 'desc': 'Fast'},
                {'name': 'Optimize', 'code': 'simplifier', 'desc': 'Reduce'},
            ]},
            {'name': 'Quantize', 'items': [
                {'name': 'Dynamic', 'code': 'quantize_dynamic', 'desc': 'Easy'},
                {'name': 'Static', 'code': 'prepare+convert', 'desc': 'Accurate'},
                {'name': 'QAT', 'code': 'Aware training', 'desc': 'Best'},
            ]},
        ]
    },
}

# Generate all tutorial images
for folder, info in tutorials.items():
    os.makedirs(folder, exist_ok=True)
    create_tutorial_image(
        info['title'], 
        info['subtitle'], 
        info['sections'], 
        f"{folder}/overview.png",
        info['color']
    )

# Create banner
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

# Background
bg = FancyBboxPatch((0.1, 0.1), 11.8, 3.8,
                     boxstyle="round,pad=0.02,rounding_size=0.3",
                     facecolor='#161b22', edgecolor='#EE4C2C',
                     linewidth=3)
ax.add_patch(bg)

# PyTorch logo area
logo_box = FancyBboxPatch((0.4, 0.8), 2.2, 2.4,
                           boxstyle="round,pad=0.02,rounding_size=0.2",
                           facecolor='#EE4C2C', edgecolor='none', alpha=0.9)
ax.add_patch(logo_box)
ax.text(1.5, 2.0, 'Py', fontsize=32, fontweight='bold', color='white', ha='center', va='center')
ax.text(1.5, 1.3, 'Torch', fontsize=14, fontweight='bold', color='white', ha='center', va='center')

# Main title
ax.text(6.8, 2.8, 'PyTorch: Zero to Advanced', fontsize=26, fontweight='bold',
        color='white', ha='center', va='center')
ax.text(6.8, 2.1, 'Complete Deep Learning Tutorial Series', fontsize=13,
        color='#8b949e', ha='center', va='center')

# Stats
stats = [
    ('12', 'Tutorials'),
    ('Colab', 'Ready'),
    ('GPU', 'Support'),
]
for i, (num, label) in enumerate(stats):
    x = 4.5 + i * 2.3
    ax.text(x, 1.1, num, fontsize=16, fontweight='bold', 
            color='#EE4C2C', ha='center', va='center')
    ax.text(x, 0.6, label, fontsize=10, color='#8b949e', ha='center', va='center')

plt.tight_layout(pad=0.3)
plt.savefig('banner.png', bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
plt.close()
print("Created: banner.png")

print("\n✅ All images generated successfully!")
