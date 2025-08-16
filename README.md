# Feedforward Neural Network (FFNN) from Scratch

A pure NumPy implementation of a feedforward neural network with support for multiple activation functions, designed for educational purposes and deep learning fundamentals.

## Features

- **Pure NumPy Implementation**: Built from scratch without high-level ML frameworks
- **Multiple Activation Functions**: Supports ReLU, Sigmoid, Tanh, and Softmax
- **Flexible Architecture**: Configurable number of layers and neurons
- **MNIST Ready**: Includes data preprocessing and visualization utilities
- **Training Metrics**: Tracks loss and accuracy during training
- **Mini-batch Training**: Efficient batch processing with shuffling

## Quick Start

```python
# Load and preprocess MNIST data
X_train, Y_train, X_test, Y_test = load_mnist_data()

# Create network: 784 inputs → 128 hidden (ReLU) → 64 hidden (ReLU) → 10 outputs
network = FFNN(
    layer_sizes=[784, 128, 64, 10],
    activations=["relu", "relu"]  # Hidden layer activations only
)

# Train the network
history = network.train(
    X_train, Y_train, X_test, Y_test,
    learning_rate=0.01,
    num_epochs=50,
    batch_size=32
)

# Visualize training progress
plot_training_history(history)
```

## Installation

```bash
pip install numpy matplotlib tensorflow seaborn
```

## Architecture

### FFNN Class

The main neural network class with the following key methods:

#### `__init__(layer_sizes, activations)`
- **layer_sizes**: List of integers defining network architecture (e.g., [784, 128, 10])
- **activations**: List of activation functions for hidden layers (output uses softmax)

#### `train(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, batch_size)`
Trains the network using mini-batch gradient descent with the following parameters:
- **learning_rate**: Step size for parameter updates (typically 0.001-0.1)
- **num_epochs**: Number of complete passes through training data
- **batch_size**: Number of samples per mini-batch (default: 32)

### Supported Activation Functions

| Function | Use Case | Formula |
|----------|----------|---------|
| **ReLU** | Hidden layers | `max(0, x)` |
| **Sigmoid** | Hidden layers | `1/(1 + e^(-x))` |
| **Tanh** | Hidden layers | `tanh(x)` |
| **Softmax** | Output layer (automatic) | `e^x / Σe^x` |

## Implementation Details

### Weight Initialization
Uses He initialization for optimal gradient flow:
```python
W = np.random.randn(n_out, n_in) * sqrt(2.0 / n_in)
```

### Loss Function
Cross-entropy loss with numerical stability:
```python
loss = -Σ(y_true * log(y_pred + ε)) / m
```

### Gradient Computation
Implements backpropagation with activation-specific derivatives:
- **ReLU**: `dZ = dA * (Z > 0)`
- **Sigmoid**: `dZ = dA * A * (1 - A)`
- **Tanh**: `dZ = dA * (1 - A²)`

## Example Usage

### Basic Classification
```python
# Simple 3-layer network
network = FFNN([784, 64, 10], ["relu"])

history = network.train(
    X_train, Y_train, X_test, Y_test,
    learning_rate=0.01,
    num_epochs=30
)
```

### Deep Network
```python
# Deeper network with mixed activations
network = FFNN(
    layer_sizes=[784, 256, 128, 64, 10],
    activations=["relu", "tanh", "relu"]
)
```

### Making Predictions
```python
# Forward pass for predictions
predictions, _ = network.forward_propagation(X_test)
predicted_classes = np.argmax(predictions, axis=0)
```

## Performance

Typical results on MNIST (784→128→64→10 architecture):
- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~96-97%
- **Training Time**: ~2-3 minutes (50 epochs)

## Data Format

### Input Requirements
- **X**: Shape `(n_features, n_samples)` - Features in rows, samples in columns
- **Y**: Shape `(n_classes, n_samples)` - One-hot encoded labels

### MNIST Preprocessing
The `load_mnist_data()` function automatically handles:
- Reshaping images from 28×28 to 784×1 vectors
- Normalization to [0,1] range
- One-hot encoding of labels
- Proper matrix transposition

## Visualization

The `plot_training_history()` function creates dual plots showing:
1. **Training Loss**: Cross-entropy loss over epochs
2. **Accuracy Comparison**: Training vs test accuracy curves

## File Structure

```
ffnn.py                 # Main implementation
├── FFNN               # Neural network class
├── load_mnist_data()  # Data preprocessing
└── plot_training_history()  # Visualization utilities
```

## Educational Value

This implementation is ideal for understanding:
- **Forward Propagation**: How data flows through network layers
- **Backpropagation**: Gradient computation and error propagation
- **Activation Functions**: Impact of different non-linearities
- **Weight Initialization**: Importance of proper parameter setup
- **Mini-batch Training**: Efficient stochastic gradient descent

## Limitations

- **Performance**: Slower than optimized frameworks (TensorFlow, PyTorch)
- **GPU Support**: CPU-only implementation
- **Advanced Features**: No regularization, dropout, or adaptive optimizers
- **Memory**: Stores all intermediate values for gradient computation

## Contributing

This is an educational implementation. For production use, consider:
- Adding regularization techniques (L1/L2, dropout)
- Implementing adaptive optimizers (Adam, RMSprop)
- Adding batch normalization
- GPU acceleration with CuPy or similar

## License

Open source - feel free to use for educational purposes.

---

*Built for learning deep learning fundamentals through hands-on implementation.*
