"""Generate block diagram style images for PyTorch tutorials."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.family'] = 'DejaVu Sans'

def draw_block(ax, x, y, w, h, text, code=None, color='#3b82f6', text_color='white'):
    """Draw a single block with text."""
    block = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(block)
    
    if code:
        ax.text(x, y + 0.15, text, fontsize=9, fontweight='bold',
                color=text_color, ha='center', va='center')
        ax.text(x, y - 0.15, code, fontsize=7, fontfamily='monospace',
                color='#fef08a', ha='center', va='center')
    else:
        ax.text(x, y, text, fontsize=9, fontweight='bold',
                color=text_color, ha='center', va='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#64748b'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

def draw_arrow_curved(ax, x1, y1, x2, y2, color='#64748b', curve='arc3,rad=0.2'):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2,
                               connectionstyle=curve))

# ============ TUTORIAL 1: BASICS ============
def create_basics_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    # Title
    ax.text(6, 5.5, 'PyTorch Basics - Setup Flow', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Flow: Install -> Import -> GPU Check -> Create Tensor -> Use
    blocks = [
        (1.5, 3.5, 'Install', 'pip install torch', '#22c55e'),
        (4, 3.5, 'Import', 'import torch', '#3b82f6'),
        (6.5, 3.5, 'GPU Check', 'cuda.is_available()', '#f59e0b'),
        (9, 3.5, 'Create Tensor', 'torch.tensor([...])', '#ec4899'),
    ]
    
    for x, y, text, code, color in blocks:
        draw_block(ax, x, y, 2.2, 1, text, code, color)
    
    # Arrows
    for i in range(len(blocks) - 1):
        draw_arrow(ax, blocks[i][0] + 1.1, blocks[i][1], 
                   blocks[i+1][0] - 1.1, blocks[i+1][1], '#94a3b8')
    
    # Bottom: Device selection
    ax.text(6.5, 2, 'Device Selection', fontsize=11, fontweight='bold', color='#94a3b8', ha='center')
    
    draw_block(ax, 4, 1, 2, 0.8, 'CPU', 'device("cpu")', '#64748b')
    draw_block(ax, 9, 1, 2, 0.8, 'GPU', 'device("cuda")', '#22c55e')
    
    draw_arrow(ax, 6.5, 3, 4, 1.4, '#94a3b8')
    draw_arrow(ax, 6.5, 3, 9, 1.4, '#94a3b8')
    
    ax.text(5.5, 1.8, 'False', fontsize=8, color='#94a3b8')
    ax.text(7.5, 1.8, 'True', fontsize=8, color='#22c55e')
    
    plt.savefig('01_basics/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 01_basics/overview.png")

# ============ TUTORIAL 2: TENSORS ============
def create_tensors_diagram():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 6.5, 'Tensor Operations Flow', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Creation methods (left)
    ax.text(1.5, 5.5, 'CREATE', fontsize=10, fontweight='bold', color='#22c55e', ha='center')
    creates = [('zeros()', 5), ('ones()', 4.5), ('randn()', 4), ('arange()', 3.5), ('eye()', 3)]
    for name, y in creates:
        draw_block(ax, 1.5, y, 1.6, 0.4, name, color='#166534')
    
    # Central tensor
    draw_block(ax, 4.5, 4, 2, 1.2, 'TENSOR', '[B, C, H, W]', '#3b82f6')
    
    # Arrows to tensor
    for _, y in creates:
        draw_arrow(ax, 2.3, y, 3.5, 4, '#22c55e')
    
    # Operations (right side)
    ax.text(7.5, 5.8, 'OPERATIONS', fontsize=10, fontweight='bold', color='#f59e0b', ha='center')
    
    # Math ops
    draw_block(ax, 7, 5, 1.4, 0.5, 'Math', '+, -, *, /', '#b45309')
    draw_block(ax, 8.5, 5, 1.4, 0.5, 'Reduce', 'sum, mean', '#b45309')
    draw_block(ax, 10, 5, 1.4, 0.5, 'Compare', '>, <, ==', '#b45309')
    
    # Linear algebra
    ax.text(8.5, 4.2, 'LINEAR ALGEBRA', fontsize=9, fontweight='bold', color='#ec4899', ha='center')
    draw_block(ax, 7, 3.5, 1.4, 0.5, 'MatMul', 'A @ B', '#9d174d')
    draw_block(ax, 8.5, 3.5, 1.4, 0.5, 'Inverse', 'linalg.inv', '#9d174d')
    draw_block(ax, 10, 3.5, 1.4, 0.5, 'SVD', 'linalg.svd', '#9d174d')
    
    # Reshape
    ax.text(8.5, 2.6, 'RESHAPE', fontsize=9, fontweight='bold', color='#8b5cf6', ha='center')
    draw_block(ax, 7, 2, 1.4, 0.5, 'view()', 'reshape', '#6d28d9')
    draw_block(ax, 8.5, 2, 1.4, 0.5, 'squeeze()', 'remove 1s', '#6d28d9')
    draw_block(ax, 10, 2, 1.4, 0.5, 'cat()', 'concat', '#6d28d9')
    
    # Arrows from tensor
    draw_arrow(ax, 5.5, 4.3, 6.3, 5, '#94a3b8')
    draw_arrow(ax, 5.5, 4, 6.3, 3.5, '#94a3b8')
    draw_arrow(ax, 5.5, 3.7, 6.3, 2, '#94a3b8')
    
    # NumPy bridge (bottom)
    ax.text(4.5, 1.5, 'NumPy Bridge', fontsize=10, fontweight='bold', color='#06b6d4', ha='center')
    draw_block(ax, 2.5, 0.8, 2, 0.6, 'NumPy Array', 'np.array()', '#0891b2')
    draw_block(ax, 6.5, 0.8, 2, 0.6, 'PyTorch Tensor', 'torch.tensor()', '#3b82f6')
    
    draw_arrow(ax, 3.5, 1, 5.5, 1, '#06b6d4')
    draw_arrow(ax, 5.5, 0.6, 3.5, 0.6, '#06b6d4')
    ax.text(4.5, 1.15, 'from_numpy()', fontsize=7, color='#06b6d4', ha='center')
    ax.text(4.5, 0.45, '.numpy()', fontsize=7, color='#06b6d4', ha='center')
    
    plt.savefig('02_tensors/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 02_tensors/overview.png")

# ============ TUTORIAL 3: AUTOGRAD ============
def create_autograd_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Autograd - Computational Graph', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Forward pass (top)
    ax.text(6, 4.8, 'Forward Pass', fontsize=11, color='#22c55e', ha='center')
    
    draw_block(ax, 2, 4, 1.8, 0.8, 'x', 'requires_grad=True', '#22c55e')
    draw_block(ax, 4.5, 4, 1.5, 0.8, 'W*x', 'Linear', '#3b82f6')
    draw_block(ax, 7, 4, 1.5, 0.8, 'ReLU', 'Activation', '#f59e0b')
    draw_block(ax, 9.5, 4, 1.5, 0.8, 'Loss', 'MSE/CE', '#ef4444')
    
    draw_arrow(ax, 2.9, 4, 3.75, 4, '#22c55e')
    draw_arrow(ax, 5.25, 4, 6.25, 4, '#22c55e')
    draw_arrow(ax, 7.75, 4, 8.75, 4, '#22c55e')
    
    # Backward pass (bottom)
    ax.text(6, 2.2, 'Backward Pass: loss.backward()', fontsize=11, color='#ec4899', ha='center')
    
    draw_block(ax, 9.5, 1.5, 1.5, 0.8, 'dL/dL', '= 1', '#ec4899')
    draw_block(ax, 7, 1.5, 1.5, 0.8, 'dL/da', 'grad_fn', '#ec4899')
    draw_block(ax, 4.5, 1.5, 1.5, 0.8, 'dL/dz', 'grad_fn', '#ec4899')
    draw_block(ax, 2, 1.5, 1.8, 0.8, 'x.grad', 'dL/dx', '#ec4899')
    
    draw_arrow(ax, 8.75, 1.5, 7.75, 1.5, '#ec4899')
    draw_arrow(ax, 6.25, 1.5, 5.25, 1.5, '#ec4899')
    draw_arrow(ax, 3.75, 1.5, 2.9, 1.5, '#ec4899')
    
    # Chain rule annotation
    ax.text(6, 0.5, 'Chain Rule: dL/dx = dL/da * da/dz * dz/dx', fontsize=10, 
            color='#94a3b8', ha='center', style='italic')
    
    plt.savefig('03_autograd/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 03_autograd/overview.png")

# ============ TUTORIAL 4: NEURAL NETWORKS ============
def create_nn_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Neural Network Architecture', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Input layer
    for i, y in enumerate([4, 3.2, 2.4]):
        circle = Circle((1.5, y), 0.25, facecolor='#22c55e', edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
    ax.text(1.5, 1.5, 'Input', fontsize=9, color='#22c55e', ha='center')
    
    # Hidden layer 1
    draw_block(ax, 4, 3.2, 1.8, 2, 'Linear\n+\nReLU', color='#3b82f6')
    ax.text(4, 1.5, 'Hidden 1', fontsize=9, color='#3b82f6', ha='center')
    
    # Hidden layer 2
    draw_block(ax, 6.5, 3.2, 1.8, 2, 'Linear\n+\nReLU', color='#8b5cf6')
    ax.text(6.5, 1.5, 'Hidden 2', fontsize=9, color='#8b5cf6', ha='center')
    
    # Output layer
    draw_block(ax, 9, 3.2, 1.8, 1.5, 'Linear\n+\nSoftmax', color='#ec4899')
    ax.text(9, 1.5, 'Output', fontsize=9, color='#ec4899', ha='center')
    
    # Output nodes
    for i, y in enumerate([4, 3.2, 2.4]):
        circle = Circle((10.8, y), 0.25, facecolor='#ef4444', edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
    ax.text(10.8, 1.5, 'Classes', fontsize=9, color='#ef4444', ha='center')
    
    # Arrows
    for y in [4, 3.2, 2.4]:
        draw_arrow(ax, 1.75, y, 3.1, 3.2, '#64748b')
    draw_arrow(ax, 4.9, 3.2, 5.6, 3.2, '#64748b')
    draw_arrow(ax, 7.4, 3.2, 8.1, 3.2, '#64748b')
    for y in [4, 3.2, 2.4]:
        draw_arrow(ax, 9.9, 3.2, 10.55, y, '#64748b')
    
    # Code annotation
    ax.text(6, 0.6, 'class Net(nn.Module):  def forward(self, x): return self.layers(x)', 
            fontsize=8, fontfamily='monospace', color='#94a3b8', ha='center')
    
    plt.savefig('04_neural_networks/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 04_neural_networks/overview.png")

# ============ TUTORIAL 5: DATA LOADING ============
def create_data_loading_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Data Loading Pipeline', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Raw data
    draw_block(ax, 1.5, 3.5, 2, 1.5, 'Raw Data', 'Images/CSV', '#64748b')
    
    # Dataset
    draw_block(ax, 4.5, 3.5, 2.2, 1.5, 'Dataset', '__getitem__(i)', '#22c55e')
    ax.text(4.5, 2.5, '__len__, transform', fontsize=7, color='#22c55e', ha='center')
    
    # Transform
    draw_block(ax, 7.5, 4.5, 2, 0.8, 'Transform', 'ToTensor()', '#f59e0b')
    draw_block(ax, 7.5, 3.5, 2, 0.8, 'Transform', 'Normalize()', '#f59e0b')
    draw_block(ax, 7.5, 2.5, 2, 0.8, 'Transform', 'Augment()', '#f59e0b')
    
    # DataLoader
    draw_block(ax, 10.5, 3.5, 2, 1.5, 'DataLoader', 'Batches', '#3b82f6')
    ax.text(10.5, 2.5, 'batch_size=32', fontsize=7, color='#3b82f6', ha='center')
    
    # Arrows
    draw_arrow(ax, 2.5, 3.5, 3.4, 3.5, '#94a3b8')
    draw_arrow(ax, 5.6, 3.5, 6.5, 3.5, '#94a3b8')
    draw_arrow(ax, 8.5, 3.5, 9.5, 3.5, '#94a3b8')
    
    # Bottom: batch output
    ax.text(6, 1.2, 'Output: for batch_x, batch_y in dataloader:', fontsize=9, 
            fontfamily='monospace', color='#94a3b8', ha='center')
    
    # Batch visualization
    for i in range(4):
        rect = FancyBboxPatch((3.5 + i*1.5, 0.3), 1.2, 0.6,
                               boxstyle="round,pad=0.02", facecolor='#1e293b',
                               edgecolor='#3b82f6', linewidth=1)
        ax.add_patch(rect)
        ax.text(4.1 + i*1.5, 0.6, f'Batch {i+1}', fontsize=7, color='white', ha='center')
    
    plt.savefig('05_data_loading/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 05_data_loading/overview.png")

# ============ TUTORIAL 6: TRAINING LOOP ============
def create_training_loop_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Training Loop', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Circular flow
    steps = [
        (3, 4, 'Data\nBatch', 'x, y = batch', '#64748b'),
        (6, 4.5, 'Forward', 'pred = model(x)', '#22c55e'),
        (9, 4, 'Loss', 'loss = criterion()', '#ef4444'),
        (9, 2, 'Backward', 'loss.backward()', '#ec4899'),
        (6, 1.5, 'Update', 'optimizer.step()', '#3b82f6'),
        (3, 2, 'Zero Grad', 'optimizer.zero_grad()', '#f59e0b'),
    ]
    
    for x, y, text, code, color in steps:
        draw_block(ax, x, y, 2.2, 1, text, code, color)
    
    # Circular arrows
    draw_arrow(ax, 4.1, 4, 4.9, 4.3, '#94a3b8')
    draw_arrow(ax, 7.1, 4.5, 7.9, 4.2, '#94a3b8')
    draw_arrow(ax, 9, 3.5, 9, 2.5, '#94a3b8')
    draw_arrow(ax, 7.9, 1.7, 7.1, 1.5, '#94a3b8')
    draw_arrow(ax, 4.9, 1.5, 4.1, 1.8, '#94a3b8')
    draw_arrow(ax, 3, 2.5, 3, 3.5, '#94a3b8')
    
    # Center text
    ax.text(6, 3, 'Repeat for\neach epoch', fontsize=10, color='#94a3b8', 
            ha='center', va='center', style='italic')
    
    plt.savefig('06_training_loop/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 06_training_loop/overview.png")

# ============ TUTORIAL 7: CNN ============
def create_cnn_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'CNN Architecture', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Input image
    rect = FancyBboxPatch((0.3, 2), 1.5, 2, boxstyle="round,pad=0.02",
                           facecolor='#22c55e', edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.05, 3, 'Image\n[3,224,224]', fontsize=8, color='white', ha='center', va='center')
    
    # Conv blocks
    conv_blocks = [
        (2.8, 'Conv1\n64', '#3b82f6'),
        (4.3, 'Pool\n/2', '#8b5cf6'),
        (5.8, 'Conv2\n128', '#3b82f6'),
        (7.3, 'Pool\n/2', '#8b5cf6'),
        (8.8, 'Conv3\n256', '#3b82f6'),
    ]
    
    h = 2
    for x, text, color in conv_blocks:
        rect = FancyBboxPatch((x, 3 - h/2), 1.2, h, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.6, 3, text, fontsize=8, color='white', ha='center', va='center')
        h = max(h * 0.85, 1)
    
    # Flatten + FC
    draw_block(ax, 10.5, 3.5, 1.2, 0.8, 'Flatten', color='#f59e0b')
    draw_block(ax, 10.5, 2.5, 1.2, 0.8, 'FC 512', color='#ec4899')
    draw_block(ax, 10.5, 1.5, 1.2, 0.8, 'FC 10', color='#ef4444')
    
    # Arrows
    positions = [1.8, 4, 5.5, 7, 8.5, 10]
    for i in range(len(positions) - 1):
        draw_arrow(ax, positions[i], 3, positions[i] + 0.7, 3, '#94a3b8')
    
    draw_arrow(ax, 10.5, 3.1, 10.5, 2.9, '#94a3b8')
    draw_arrow(ax, 10.5, 2.1, 10.5, 1.9, '#94a3b8')
    
    # Labels
    ax.text(3.4, 4.7, 'Feature Extraction', fontsize=10, color='#3b82f6', ha='center')
    ax.text(10.5, 4.7, 'Classification', fontsize=10, color='#ec4899', ha='center')
    
    plt.savefig('07_cnn/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 07_cnn/overview.png")

# ============ TUTORIAL 8: RNN/LSTM ============
def create_rnn_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'LSTM Cell Architecture', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # LSTM cell (center)
    cell = FancyBboxPatch((3.5, 1.5), 5, 3.5, boxstyle="round,pad=0.05",
                           facecolor='#1e293b', edgecolor='#3b82f6', linewidth=2)
    ax.add_patch(cell)
    
    # Gates inside
    draw_block(ax, 4.5, 4, 1.2, 0.7, 'Forget', 'sigmoid', '#ef4444')
    draw_block(ax, 6, 4, 1.2, 0.7, 'Input', 'sigmoid', '#22c55e')
    draw_block(ax, 7.5, 4, 1.2, 0.7, 'Output', 'sigmoid', '#3b82f6')
    
    # Cell state
    draw_block(ax, 6, 2.8, 2, 0.7, 'Cell State', 'c_t', '#f59e0b')
    
    # Hidden state
    draw_block(ax, 6, 1.8, 2, 0.7, 'Hidden', 'h_t', '#ec4899')
    
    # Inputs
    draw_block(ax, 1.5, 3, 1.5, 0.8, 'x_t', 'input', '#64748b')
    draw_block(ax, 1.5, 2, 1.5, 0.8, 'h_t-1', 'prev hidden', '#ec4899')
    
    # Output
    draw_block(ax, 10.5, 3, 1.5, 0.8, 'h_t', 'output', '#ec4899')
    draw_block(ax, 10.5, 2, 1.5, 0.8, 'c_t', 'cell', '#f59e0b')
    
    # Arrows
    draw_arrow(ax, 2.25, 3, 3.5, 3, '#94a3b8')
    draw_arrow(ax, 2.25, 2, 3.5, 2.5, '#94a3b8')
    draw_arrow(ax, 8.5, 3, 9.75, 3, '#94a3b8')
    draw_arrow(ax, 8.5, 2.5, 9.75, 2, '#94a3b8')
    
    # Formula
    ax.text(6, 0.8, 'h_t = o_t * tanh(c_t)    c_t = f_t * c_t-1 + i_t * g_t', 
            fontsize=9, fontfamily='monospace', color='#94a3b8', ha='center')
    
    plt.savefig('08_rnn_lstm/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 08_rnn_lstm/overview.png")

# ============ TUTORIAL 9: TRANSFORMERS ============
def create_transformer_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Self-Attention Mechanism', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Input
    draw_block(ax, 1.5, 3, 1.5, 1, 'Input\nX', color='#64748b')
    
    # Q, K, V projections
    draw_block(ax, 4, 4.2, 1.2, 0.7, 'Query', 'X @ W_q', '#ef4444')
    draw_block(ax, 4, 3, 1.2, 0.7, 'Key', 'X @ W_k', '#22c55e')
    draw_block(ax, 4, 1.8, 1.2, 0.7, 'Value', 'X @ W_v', '#3b82f6')
    
    # Attention
    draw_block(ax, 6.5, 3.5, 2, 1, 'Attention\nScore', 'softmax(QK/d)', '#f59e0b')
    
    # Output
    draw_block(ax, 9, 3, 1.8, 1, 'Output', 'Score @ V', '#ec4899')
    
    # Arrows
    draw_arrow(ax, 2.25, 3.3, 3.4, 4.2, '#94a3b8')
    draw_arrow(ax, 2.25, 3, 3.4, 3, '#94a3b8')
    draw_arrow(ax, 2.25, 2.7, 3.4, 1.8, '#94a3b8')
    
    draw_arrow(ax, 4.6, 4.2, 5.5, 3.8, '#ef4444')
    draw_arrow(ax, 4.6, 3, 5.5, 3.3, '#22c55e')
    
    draw_arrow(ax, 7.5, 3.2, 8.1, 3, '#f59e0b')
    draw_arrow(ax, 4.6, 1.8, 8.1, 2.7, '#3b82f6')
    
    # Multi-head annotation
    ax.text(6, 1, 'Multi-Head: Concat(head_1, ..., head_h) @ W_o', fontsize=9,
            fontfamily='monospace', color='#94a3b8', ha='center')
    
    # Formula
    ax.text(6.5, 2.3, 'Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V', fontsize=8,
            fontfamily='monospace', color='#f59e0b', ha='center')
    
    plt.savefig('09_transformers/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 09_transformers/overview.png")

# ============ TUTORIAL 10: TRANSFER LEARNING ============
def create_transfer_learning_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Transfer Learning Pipeline', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Pretrained model
    draw_block(ax, 2, 4, 2.5, 1.2, 'Pretrained\nResNet/BERT', color='#3b82f6')
    ax.text(2, 3.1, 'ImageNet/Wikipedia', fontsize=7, color='#94a3b8', ha='center')
    
    # Freeze backbone
    backbone = FancyBboxPatch((4, 2.5), 3, 2.5, boxstyle="round,pad=0.03",
                               facecolor='#1e293b', edgecolor='#22c55e', linewidth=2)
    ax.add_patch(backbone)
    ax.text(5.5, 4.5, 'Frozen Backbone', fontsize=9, color='#22c55e', ha='center')
    ax.text(5.5, 3.8, 'requires_grad=False', fontsize=7, fontfamily='monospace', 
            color='#22c55e', ha='center')
    ax.text(5.5, 3, 'Feature\nExtractor', fontsize=10, color='white', ha='center')
    
    # New head
    draw_block(ax, 8.5, 3.5, 2, 1, 'New Head', 'nn.Linear()', '#ec4899')
    ax.text(8.5, 2.7, 'Your classes', fontsize=7, color='#94a3b8', ha='center')
    
    # Your data
    draw_block(ax, 10.5, 1.5, 1.8, 1, 'Your\nDataset', color='#f59e0b')
    
    # Arrows
    draw_arrow(ax, 3.25, 4, 4, 3.5, '#94a3b8')
    draw_arrow(ax, 7, 3.5, 7.5, 3.5, '#94a3b8')
    draw_arrow(ax, 10.5, 2, 10.5, 2.5, '#94a3b8')
    draw_arrow_curved(ax, 10.5, 2.5, 8.5, 2.7, '#f59e0b', 'arc3,rad=-0.3')
    
    # Fine-tune settings
    ax.text(6, 1, 'Fine-tune: lr=1e-5, epochs=5-10, small dataset OK', fontsize=9,
            fontfamily='monospace', color='#94a3b8', ha='center')
    
    plt.savefig('10_transfer_learning/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 10_transfer_learning/overview.png")

# ============ TUTORIAL 11: GAN ============
def create_gan_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'GAN Architecture', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # Noise
    draw_block(ax, 1.5, 4, 1.5, 1, 'Noise z', 'N(0,1)', '#64748b')
    
    # Generator
    gen = FancyBboxPatch((3.2, 3), 2.5, 2, boxstyle="round,pad=0.03",
                          facecolor='#1e293b', edgecolor='#22c55e', linewidth=2)
    ax.add_patch(gen)
    ax.text(4.45, 4.5, 'Generator', fontsize=10, fontweight='bold', color='#22c55e', ha='center')
    ax.text(4.45, 3.5, 'ConvT\nUpsample', fontsize=9, color='white', ha='center')
    
    # Fake image
    draw_block(ax, 6.8, 4, 1.3, 1, 'Fake\nImage', color='#f59e0b')
    
    # Real image
    draw_block(ax, 6.8, 2, 1.3, 1, 'Real\nImage', color='#3b82f6')
    
    # Discriminator
    disc = FancyBboxPatch((8.3, 2), 2.5, 3, boxstyle="round,pad=0.03",
                           facecolor='#1e293b', edgecolor='#ef4444', linewidth=2)
    ax.add_patch(disc)
    ax.text(9.55, 4.5, 'Discriminator', fontsize=10, fontweight='bold', color='#ef4444', ha='center')
    ax.text(9.55, 3.2, 'Conv\nDownsample', fontsize=9, color='white', ha='center')
    
    # Output
    draw_block(ax, 9.55, 1.2, 1.5, 0.6, 'Real/Fake', '0 or 1', '#ec4899')
    
    # Arrows
    draw_arrow(ax, 2.25, 4, 3.2, 4, '#94a3b8')
    draw_arrow(ax, 5.7, 4, 6.15, 4, '#22c55e')
    draw_arrow(ax, 7.45, 4, 8.3, 3.8, '#f59e0b')
    draw_arrow(ax, 7.45, 2, 8.3, 2.5, '#3b82f6')
    draw_arrow(ax, 9.55, 2, 9.55, 1.5, '#94a3b8')
    
    # Loss annotations
    ax.text(2, 1, 'G Loss: fool D (make D output 1 for fakes)', fontsize=8,
            color='#22c55e', ha='left')
    ax.text(2, 0.5, 'D Loss: detect fakes (output 1 for real, 0 for fake)', fontsize=8,
            color='#ef4444', ha='left')
    
    plt.savefig('11_gan/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 11_gan/overview.png")

# ============ TUTORIAL 12: DEPLOYMENT ============
def create_deployment_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    ax.text(6, 5.5, 'Model Deployment Pipeline', fontsize=16, fontweight='bold',
            color='white', ha='center')
    
    # PyTorch model
    draw_block(ax, 1.5, 3.5, 2, 1.2, 'PyTorch\nModel', color='#3b82f6')
    ax.text(1.5, 2.6, 'model.pth', fontsize=7, color='#94a3b8', ha='center')
    
    # Export options
    draw_block(ax, 4.5, 4.5, 2, 0.9, 'TorchScript', 'jit.trace()', '#22c55e')
    draw_block(ax, 4.5, 3.2, 2, 0.9, 'ONNX', 'onnx.export()', '#f59e0b')
    draw_block(ax, 4.5, 1.9, 2, 0.9, 'Quantize', 'INT8', '#ec4899')
    
    # Runtimes
    draw_block(ax, 7.5, 4.5, 2, 0.9, 'LibTorch', 'C++', '#22c55e')
    draw_block(ax, 7.5, 3.2, 2, 0.9, 'ONNX RT', 'Universal', '#f59e0b')
    draw_block(ax, 7.5, 1.9, 2, 0.9, 'Mobile', 'Lite', '#ec4899')
    
    # Deployment targets
    draw_block(ax, 10.5, 4.5, 1.8, 0.9, 'Server', color='#3b82f6')
    draw_block(ax, 10.5, 3.2, 1.8, 0.9, 'Edge', color='#8b5cf6')
    draw_block(ax, 10.5, 1.9, 1.8, 0.9, 'Mobile', color='#64748b')
    
    # Arrows
    draw_arrow(ax, 2.5, 4, 3.5, 4.5, '#94a3b8')
    draw_arrow(ax, 2.5, 3.5, 3.5, 3.2, '#94a3b8')
    draw_arrow(ax, 2.5, 3, 3.5, 1.9, '#94a3b8')
    
    for y in [4.5, 3.2, 1.9]:
        draw_arrow(ax, 5.5, y, 6.5, y, '#94a3b8')
        draw_arrow(ax, 8.5, y, 9.6, y, '#94a3b8')
    
    # Speed annotation
    ax.text(6, 0.8, 'Optimization: 2-4x faster inference, 75% smaller model', fontsize=9,
            color='#94a3b8', ha='center')
    
    plt.savefig('12_deployment/overview.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: 12_deployment/overview.png")

# ============ BANNER ============
def create_banner():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    
    # Background
    bg = FancyBboxPatch((0.1, 0.1), 11.8, 3.8,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor='#1e293b', edgecolor='#EE4C2C',
                         linewidth=3)
    ax.add_patch(bg)
    
    # PyTorch logo representation
    logo = FancyBboxPatch((0.5, 1), 2, 2,
                           boxstyle="round,pad=0.02,rounding_size=0.2",
                           facecolor='#EE4C2C', edgecolor='none')
    ax.add_patch(logo)
    ax.text(1.5, 2.3, 'Py', fontsize=28, fontweight='bold', color='white', ha='center')
    ax.text(1.5, 1.5, 'Torch', fontsize=12, fontweight='bold', color='white', ha='center')
    
    # Title
    ax.text(6.5, 2.8, 'PyTorch: Zero to Advanced', fontsize=24, fontweight='bold',
            color='white', ha='center')
    ax.text(6.5, 2, 'Complete Deep Learning Tutorial', fontsize=12,
            color='#94a3b8', ha='center')
    
    # Stats boxes
    stats = [
        (4, '12', 'Tutorials', '#22c55e'),
        (6.5, 'GPU', 'Ready', '#f59e0b'),
        (9, 'Colab', 'Notebooks', '#3b82f6'),
    ]
    for x, num, label, color in stats:
        box = FancyBboxPatch((x - 0.8, 0.4), 1.6, 1,
                              boxstyle="round,pad=0.02", facecolor='#0f172a',
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 1.05, num, fontsize=14, fontweight='bold', color=color, ha='center')
        ax.text(x, 0.65, label, fontsize=8, color='#94a3b8', ha='center')
    
    plt.savefig('banner.png', bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print("Created: banner.png")

# Generate all images
if __name__ == '__main__':
    os.makedirs('01_basics', exist_ok=True)
    os.makedirs('02_tensors', exist_ok=True)
    os.makedirs('03_autograd', exist_ok=True)
    os.makedirs('04_neural_networks', exist_ok=True)
    os.makedirs('05_data_loading', exist_ok=True)
    os.makedirs('06_training_loop', exist_ok=True)
    os.makedirs('07_cnn', exist_ok=True)
    os.makedirs('08_rnn_lstm', exist_ok=True)
    os.makedirs('09_transformers', exist_ok=True)
    os.makedirs('10_transfer_learning', exist_ok=True)
    os.makedirs('11_gan', exist_ok=True)
    os.makedirs('12_deployment', exist_ok=True)
    
    create_basics_diagram()
    create_tensors_diagram()
    create_autograd_diagram()
    create_nn_diagram()
    create_data_loading_diagram()
    create_training_loop_diagram()
    create_cnn_diagram()
    create_rnn_diagram()
    create_transformer_diagram()
    create_transfer_learning_diagram()
    create_gan_diagram()
    create_deployment_diagram()
    create_banner()
    
    print("\n✅ All block diagram images generated!")
