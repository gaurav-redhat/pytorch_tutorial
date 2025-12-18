"""Generate detailed tutorial images for PyTorch Zero to Advanced."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_detailed_image(title, sections, filename, color='#EE4C2C'):
    """Create a detailed tutorial image with multiple sections."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Background
    bg = patches.FancyBboxPatch((0.1, 0.1), 13.8, 7.8, 
                                  boxstyle="round,pad=0.05",
                                  facecolor='#1a1a2e', edgecolor=color, linewidth=3)
    ax.add_patch(bg)
    
    # Title
    ax.text(7, 7.3, title, fontsize=22, fontweight='bold', 
            color='white', ha='center', va='center')
    
    # Draw sections
    n_sections = len(sections)
    section_height = 5.5 / n_sections
    
    for i, section in enumerate(sections):
        y_start = 6.5 - (i + 1) * section_height
        
        # Section header
        header_box = patches.FancyBboxPatch((0.3, y_start + section_height - 0.6), 
                                             3, 0.5,
                                             boxstyle="round,pad=0.02",
                                             facecolor=color, edgecolor='white', 
                                             linewidth=1, alpha=0.9)
        ax.add_patch(header_box)
        ax.text(1.8, y_start + section_height - 0.35, section['name'], 
                fontsize=11, fontweight='bold', color='white', ha='center')
        
        # Section content boxes
        items = section.get('items', [])
        n_items = len(items)
        if n_items > 0:
            item_width = 10 / n_items
            for j, item in enumerate(items):
                x = 3.5 + j * item_width + item_width/2
                
                # Item box
                item_box = patches.FancyBboxPatch((x - item_width/2 + 0.1, y_start + 0.1), 
                                                  item_width - 0.2, section_height - 0.8,
                                                  boxstyle="round,pad=0.02",
                                                  facecolor='#16213e', edgecolor=color, 
                                                  linewidth=1, alpha=0.8)
                ax.add_patch(item_box)
                
                # Item name
                ax.text(x, y_start + section_height - 1.0, item['name'], 
                        fontsize=9, fontweight='bold', color=color, ha='center')
                
                # Item code/desc
                if 'code' in item:
                    ax.text(x, y_start + section_height/2 - 0.2, item['code'], 
                            fontsize=8, fontfamily='monospace', color='#00ff88', ha='center')
                if 'desc' in item:
                    ax.text(x, y_start + 0.3, item['desc'], 
                            fontsize=7, color='#aaa', ha='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()

# Detailed tutorial images
tutorials = {
    '01_basics': {
        'title': '01. PyTorch Basics',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Setup', 'items': [
                {'name': 'Install', 'code': 'pip install torch', 'desc': 'CPU/GPU'},
                {'name': 'Import', 'code': 'import torch', 'desc': 'Main module'},
                {'name': 'Version', 'code': 'torch.__version__', 'desc': 'Check version'},
            ]},
            {'name': 'GPU', 'items': [
                {'name': 'Check CUDA', 'code': 'torch.cuda.is_available()', 'desc': 'True/False'},
                {'name': 'Device', 'code': "device('cuda')", 'desc': 'GPU device'},
                {'name': 'Move', 'code': 'x.to(device)', 'desc': 'CPU <-> GPU'},
            ]},
            {'name': 'First Tensor', 'items': [
                {'name': 'Create', 'code': 'torch.tensor([1,2,3])', 'desc': 'From list'},
                {'name': 'Shape', 'code': 'x.shape', 'desc': 'Dimensions'},
                {'name': 'Dtype', 'code': 'x.dtype', 'desc': 'Data type'},
            ]},
        ]
    },
    '02_tensors': {
        'title': '02. Tensors - Complete Operations',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Creation', 'items': [
                {'name': 'zeros/ones', 'code': 'torch.zeros(3,4)', 'desc': 'Fill values'},
                {'name': 'rand/randn', 'code': 'torch.randn(3,3)', 'desc': 'Random'},
                {'name': 'arange', 'code': 'torch.arange(0,10,2)', 'desc': 'Range'},
                {'name': 'linspace', 'code': 'torch.linspace(0,1,10)', 'desc': 'Linear'},
                {'name': 'eye', 'code': 'torch.eye(3)', 'desc': 'Identity'},
            ]},
            {'name': 'NumPy Bridge', 'items': [
                {'name': 'To NumPy', 'code': 'x.numpy()', 'desc': 'Tensor->Array'},
                {'name': 'From NumPy', 'code': 'torch.from_numpy(arr)', 'desc': 'Array->Tensor'},
                {'name': 'Share Memory', 'code': 'x.numpy() shares', 'desc': 'Same data!'},
            ]},
            {'name': 'Linear Algebra', 'items': [
                {'name': 'MatMul', 'code': 'A @ B, torch.mm()', 'desc': 'Matrix multiply'},
                {'name': 'Transpose', 'code': 'x.T, x.transpose()', 'desc': 'Swap dims'},
                {'name': 'Inverse', 'code': 'torch.linalg.inv()', 'desc': 'Matrix inverse'},
                {'name': 'SVD', 'code': 'torch.linalg.svd()', 'desc': 'Decomposition'},
                {'name': 'Eigen', 'code': 'torch.linalg.eig()', 'desc': 'Eigenvalues'},
            ]},
            {'name': 'Reshape & Index', 'items': [
                {'name': 'view/reshape', 'code': 'x.view(3,4)', 'desc': 'Change shape'},
                {'name': 'squeeze', 'code': 'x.squeeze()', 'desc': 'Remove dim=1'},
                {'name': 'unsqueeze', 'code': 'x.unsqueeze(0)', 'desc': 'Add dim'},
                {'name': 'slice', 'code': 'x[0:2, 1:]', 'desc': 'Select range'},
                {'name': 'boolean', 'code': 'x[x > 0]', 'desc': 'Filter'},
            ]},
        ]
    },
    '03_autograd': {
        'title': '03. Autograd - Automatic Differentiation',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Gradient Tracking', 'items': [
                {'name': 'Enable', 'code': 'requires_grad=True', 'desc': 'Track ops'},
                {'name': 'Backward', 'code': 'loss.backward()', 'desc': 'Compute grads'},
                {'name': 'Access', 'code': 'x.grad', 'desc': 'Get gradient'},
                {'name': 'Zero', 'code': 'x.grad.zero_()', 'desc': 'Reset grads'},
            ]},
            {'name': 'Computational Graph', 'items': [
                {'name': 'Leaf', 'code': 'x (input)', 'desc': 'Start node'},
                {'name': 'Intermediate', 'code': 'y = f(x)', 'desc': 'Operations'},
                {'name': 'Output', 'code': 'loss', 'desc': 'End node'},
                {'name': 'grad_fn', 'code': 'y.grad_fn', 'desc': 'Track history'},
            ]},
            {'name': 'Control', 'items': [
                {'name': 'no_grad', 'code': 'with torch.no_grad():', 'desc': 'Disable'},
                {'name': 'detach', 'code': 'x.detach()', 'desc': 'Remove from graph'},
                {'name': 'retain', 'code': 'retain_graph=True', 'desc': 'Keep graph'},
            ]},
        ]
    },
    '04_neural_networks': {
        'title': '04. Neural Networks - nn.Module',
        'color': '#4CAF50',
        'sections': [
            {'name': 'Building Blocks', 'items': [
                {'name': 'Linear', 'code': 'nn.Linear(in, out)', 'desc': 'Dense layer'},
                {'name': 'Conv2d', 'code': 'nn.Conv2d(c,c,k)', 'desc': 'Convolution'},
                {'name': 'BatchNorm', 'code': 'nn.BatchNorm2d(c)', 'desc': 'Normalize'},
                {'name': 'Dropout', 'code': 'nn.Dropout(0.5)', 'desc': 'Regularize'},
            ]},
            {'name': 'Activations', 'items': [
                {'name': 'ReLU', 'code': 'nn.ReLU()', 'desc': 'max(0,x)'},
                {'name': 'Sigmoid', 'code': 'nn.Sigmoid()', 'desc': '1/(1+e^-x)'},
                {'name': 'Tanh', 'code': 'nn.Tanh()', 'desc': '[-1, 1]'},
                {'name': 'Softmax', 'code': 'nn.Softmax(dim=1)', 'desc': 'Probabilities'},
            ]},
            {'name': 'Model', 'items': [
                {'name': '__init__', 'code': 'super().__init__()', 'desc': 'Define layers'},
                {'name': 'forward', 'code': 'def forward(x):', 'desc': 'Computation'},
                {'name': 'parameters', 'code': 'model.parameters()', 'desc': 'Weights'},
            ]},
        ]
    },
    '05_data_loading': {
        'title': '05. Data Loading - Dataset & DataLoader',
        'color': '#FF9800',
        'sections': [
            {'name': 'Dataset', 'items': [
                {'name': '__len__', 'code': 'return len(data)', 'desc': 'Size'},
                {'name': '__getitem__', 'code': 'return x, y', 'desc': 'Get sample'},
                {'name': 'transform', 'code': 'self.transform(x)', 'desc': 'Preprocess'},
            ]},
            {'name': 'DataLoader', 'items': [
                {'name': 'batch_size', 'code': 'batch_size=32', 'desc': 'Samples/batch'},
                {'name': 'shuffle', 'code': 'shuffle=True', 'desc': 'Randomize'},
                {'name': 'num_workers', 'code': 'num_workers=4', 'desc': 'Parallel'},
                {'name': 'pin_memory', 'code': 'pin_memory=True', 'desc': 'GPU speed'},
            ]},
            {'name': 'Transforms', 'items': [
                {'name': 'ToTensor', 'code': 'ToTensor()', 'desc': 'PIL->Tensor'},
                {'name': 'Normalize', 'code': 'Normalize(m,s)', 'desc': 'Standardize'},
                {'name': 'RandomCrop', 'code': 'RandomCrop(224)', 'desc': 'Augment'},
                {'name': 'Compose', 'code': 'Compose([...])', 'desc': 'Chain'},
            ]},
        ]
    },
    '06_training_loop': {
        'title': '06. Training Loop - The Complete Workflow',
        'color': '#FF9800',
        'sections': [
            {'name': 'Forward Pass', 'items': [
                {'name': 'Input', 'code': 'x, y = batch', 'desc': 'Get data'},
                {'name': 'Predict', 'code': 'out = model(x)', 'desc': 'Forward'},
                {'name': 'Loss', 'code': 'loss = criterion(out,y)', 'desc': 'Error'},
            ]},
            {'name': 'Backward Pass', 'items': [
                {'name': 'Zero', 'code': 'optimizer.zero_grad()', 'desc': 'Clear grads'},
                {'name': 'Backprop', 'code': 'loss.backward()', 'desc': 'Compute grads'},
                {'name': 'Update', 'code': 'optimizer.step()', 'desc': 'Update weights'},
            ]},
            {'name': 'Losses', 'items': [
                {'name': 'CrossEntropy', 'code': 'nn.CrossEntropyLoss()', 'desc': 'Classification'},
                {'name': 'MSE', 'code': 'nn.MSELoss()', 'desc': 'Regression'},
                {'name': 'BCE', 'code': 'nn.BCELoss()', 'desc': 'Binary'},
            ]},
            {'name': 'Optimizers', 'items': [
                {'name': 'SGD', 'code': 'optim.SGD(lr=0.01)', 'desc': 'Basic'},
                {'name': 'Adam', 'code': 'optim.Adam(lr=0.001)', 'desc': 'Popular'},
                {'name': 'AdamW', 'code': 'optim.AdamW()', 'desc': 'With decay'},
            ]},
        ]
    },
    '07_cnn': {
        'title': '07. CNNs - Convolutional Neural Networks',
        'color': '#FF9800',
        'sections': [
            {'name': 'Convolution', 'items': [
                {'name': 'Conv2d', 'code': 'Conv2d(3,64,3,p=1)', 'desc': 'in,out,kernel'},
                {'name': 'Stride', 'code': 'stride=2', 'desc': 'Step size'},
                {'name': 'Padding', 'code': 'padding=1', 'desc': 'Border'},
                {'name': 'Dilation', 'code': 'dilation=2', 'desc': 'Gaps'},
            ]},
            {'name': 'Pooling', 'items': [
                {'name': 'MaxPool', 'code': 'MaxPool2d(2)', 'desc': 'Max value'},
                {'name': 'AvgPool', 'code': 'AvgPool2d(2)', 'desc': 'Average'},
                {'name': 'Adaptive', 'code': 'AdaptiveAvgPool2d(1)', 'desc': 'Fixed output'},
            ]},
            {'name': 'Architecture', 'items': [
                {'name': 'Input', 'code': '[B,C,H,W]', 'desc': 'Image tensor'},
                {'name': 'Conv+ReLU+Pool', 'code': 'Feature extract', 'desc': 'Repeat N times'},
                {'name': 'Flatten+FC', 'code': 'Classifier', 'desc': 'To classes'},
            ]},
        ]
    },
    '08_rnn_lstm': {
        'title': '08. RNNs & LSTMs - Sequential Data',
        'color': '#FF9800',
        'sections': [
            {'name': 'RNN Cell', 'items': [
                {'name': 'Input', 'code': 'x_t', 'desc': 'Current input'},
                {'name': 'Hidden', 'code': 'h_t-1', 'desc': 'Previous state'},
                {'name': 'Output', 'code': 'h_t = tanh(Wx+Uh)', 'desc': 'New state'},
            ]},
            {'name': 'LSTM', 'items': [
                {'name': 'Forget', 'code': 'f_t = sigmoid(...)', 'desc': 'What to forget'},
                {'name': 'Input', 'code': 'i_t = sigmoid(...)', 'desc': 'What to add'},
                {'name': 'Cell', 'code': 'c_t = f*c + i*g', 'desc': 'Memory'},
                {'name': 'Output', 'code': 'h_t = o * tanh(c)', 'desc': 'Hidden'},
            ]},
            {'name': 'PyTorch', 'items': [
                {'name': 'LSTM', 'code': 'nn.LSTM(in,h,layers)', 'desc': 'LSTM layer'},
                {'name': 'GRU', 'code': 'nn.GRU(in,h,layers)', 'desc': 'Simpler'},
                {'name': 'batch_first', 'code': 'batch_first=True', 'desc': '[B,T,F]'},
                {'name': 'bidirectional', 'code': 'bidirectional=True', 'desc': 'Both dirs'},
            ]},
        ]
    },
    '09_transformers': {
        'title': '09. Transformers - Self-Attention',
        'color': '#F44336',
        'sections': [
            {'name': 'Attention', 'items': [
                {'name': 'Query', 'code': 'Q = xW_q', 'desc': 'What to find'},
                {'name': 'Key', 'code': 'K = xW_k', 'desc': 'What I have'},
                {'name': 'Value', 'code': 'V = xW_v', 'desc': 'What to return'},
                {'name': 'Score', 'code': 'softmax(QK^T/sqrt(d))', 'desc': 'Attention'},
            ]},
            {'name': 'Multi-Head', 'items': [
                {'name': 'Split', 'code': 'head_i = Attn(QW,KW,VW)', 'desc': 'Per head'},
                {'name': 'Concat', 'code': 'Concat(head_1..h)', 'desc': 'Combine'},
                {'name': 'Project', 'code': 'output @ W_o', 'desc': 'Final'},
            ]},
            {'name': 'Position', 'items': [
                {'name': 'Sinusoidal', 'code': 'sin(pos/10000^(2i/d))', 'desc': 'Fixed'},
                {'name': 'Learned', 'code': 'nn.Embedding(max,d)', 'desc': 'Trainable'},
                {'name': 'RoPE', 'code': 'Rotary encoding', 'desc': 'Modern'},
            ]},
        ]
    },
    '10_transfer_learning': {
        'title': '10. Transfer Learning - Fine-tuning',
        'color': '#F44336',
        'sections': [
            {'name': 'Load Pretrained', 'items': [
                {'name': 'ResNet', 'code': 'resnet50(weights=...)', 'desc': 'ImageNet'},
                {'name': 'BERT', 'code': 'from_pretrained(...)', 'desc': 'HuggingFace'},
                {'name': 'ViT', 'code': 'vit_b_16(weights=...)', 'desc': 'Vision'},
            ]},
            {'name': 'Freeze', 'items': [
                {'name': 'All', 'code': 'requires_grad=False', 'desc': 'Freeze all'},
                {'name': 'Backbone', 'code': 'for p in model.features', 'desc': 'Keep encoder'},
                {'name': 'Gradual', 'code': 'Unfreeze layers', 'desc': 'Stage-wise'},
            ]},
            {'name': 'Fine-tune', 'items': [
                {'name': 'Replace head', 'code': 'model.fc = nn.Linear()', 'desc': 'New classifier'},
                {'name': 'Lower LR', 'code': 'lr=1e-5', 'desc': 'Small updates'},
                {'name': 'Train', 'code': 'Standard loop', 'desc': 'Your data'},
            ]},
        ]
    },
    '11_gan': {
        'title': '11. GANs - Generative Adversarial Networks',
        'color': '#F44336',
        'sections': [
            {'name': 'Generator', 'items': [
                {'name': 'Input', 'code': 'z ~ N(0,1)', 'desc': 'Random noise'},
                {'name': 'Upsample', 'code': 'ConvTranspose2d', 'desc': 'Grow image'},
                {'name': 'Output', 'code': 'Tanh()', 'desc': 'Fake image'},
            ]},
            {'name': 'Discriminator', 'items': [
                {'name': 'Input', 'code': 'Real or Fake image', 'desc': 'Image'},
                {'name': 'Downsample', 'code': 'Conv2d + stride', 'desc': 'Shrink'},
                {'name': 'Output', 'code': 'Sigmoid() -> 0/1', 'desc': 'Real prob'},
            ]},
            {'name': 'Training', 'items': [
                {'name': 'D loss', 'code': 'BCE(D(real),1)+BCE(D(fake),0)', 'desc': 'Detect'},
                {'name': 'G loss', 'code': 'BCE(D(G(z)),1)', 'desc': 'Fool D'},
                {'name': 'Alternate', 'code': 'Train D, then G', 'desc': 'Each step'},
            ]},
        ]
    },
    '12_deployment': {
        'title': '12. Deployment - Production',
        'color': '#F44336',
        'sections': [
            {'name': 'TorchScript', 'items': [
                {'name': 'Trace', 'code': 'torch.jit.trace(model,x)', 'desc': 'Record ops'},
                {'name': 'Script', 'code': 'torch.jit.script(model)', 'desc': 'Full compile'},
                {'name': 'Save', 'code': 'model.save("m.pt")', 'desc': 'Standalone'},
            ]},
            {'name': 'ONNX', 'items': [
                {'name': 'Export', 'code': 'torch.onnx.export(...)', 'desc': 'Universal format'},
                {'name': 'Runtime', 'code': 'onnxruntime', 'desc': 'Fast inference'},
                {'name': 'Optimize', 'code': 'onnx-simplifier', 'desc': 'Reduce ops'},
            ]},
            {'name': 'Quantization', 'items': [
                {'name': 'Dynamic', 'code': 'quantize_dynamic()', 'desc': 'Easy, FP32->INT8'},
                {'name': 'Static', 'code': 'prepare+convert', 'desc': 'Calibrated'},
                {'name': 'QAT', 'code': 'Quantization-aware', 'desc': 'Best accuracy'},
            ]},
        ]
    },
}

# Generate images
for folder, info in tutorials.items():
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/overview.png"
    create_detailed_image(info['title'], info['sections'], filename, info['color'])
    print(f"Created {filename}")

# Create improved banner
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

bg = patches.FancyBboxPatch((0.1, 0.1), 13.8, 4.8,
                              boxstyle="round,pad=0.05",
                              facecolor='#1a1a2e', edgecolor='#EE4C2C', 
                              linewidth=4)
ax.add_patch(bg)

# PyTorch flame icon representation
flame_color = '#EE4C2C'
for i, (x, y, s) in enumerate([(2, 2.5, 0.8), (2.3, 2.8, 0.6), (1.7, 2.7, 0.5)]):
    circle = patches.Circle((x, y), s, facecolor=flame_color, alpha=0.8-i*0.2)
    ax.add_patch(circle)

# Title
ax.text(7.5, 3.5, 'PyTorch: Zero to Advanced', fontsize=32, fontweight='bold',
        color='white', ha='center', va='center')
ax.text(7.5, 2.5, 'Complete Deep Learning Tutorial Series', fontsize=16,
        color='#EE4C2C', ha='center', va='center')

# Stats
ax.text(4.5, 1.2, '12 Tutorials', fontsize=12, color='#4CAF50', ha='center', fontweight='bold')
ax.text(7.5, 1.2, 'Colab Ready', fontsize=12, color='#FF9800', ha='center', fontweight='bold')
ax.text(10.5, 1.2, 'Beginner to Expert', fontsize=12, color='#F44336', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('banner.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Created banner.png")

print("\nAll detailed images generated!")
