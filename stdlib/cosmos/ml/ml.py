"""
cosmos.ml — Machine learning constellation
Activation functions, loss functions, optimizers
All return dimensionless  (Float[1]) at the type level.
"""

import numpy as np
from typing import Callable, Tuple

__all__ = [
    'mse', 'cross_entropy', 'binary_cross_entropy',
    'relu', 'sigmoid', 'softmax', 'tanh', 'gelu',
    'adam', 'sgd',
    'linear', 'conv1d', 'batch_norm', 'dropout'
]

# ============= Loss Functions =============

def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean squared error loss.
    
    Args:
        pred: Predicted values
        target: Target values
    
    Returns:
        MSE loss (Float[1] in NOVA)
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    
    return float(np.mean((pred - target) ** 2))


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Cross-entropy loss with softmax.
    
    Args:
        logits: Model outputs (unnormalized)
        labels: Target class indices
    
    Returns:
        Cross-entropy loss (Float[1] in NOVA)
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    
    # Numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Cross-entropy: -log(p[true_class])
    batch_size = logits.shape[0]
    loss = -np.log(probs[np.arange(batch_size), labels] + 1e-15)
    
    return float(np.mean(loss))


def binary_cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Binary cross-entropy loss.
    
    Args:
        pred: Predicted probabilities [0, 1]
        target: Target binary labels [0, 1]
    
    Returns:
        BCE loss (Float[1] in NOVA)
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    
    pred = np.clip(pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    loss = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
    
    return float(np.mean(loss))


# ============= Activation Functions =============

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: max(0, x)
    
    Args:
        x: Input tensor
    
    Returns:
        ReLU output (same shape)
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor
    
    Returns:
        Sigmoid output in (0, 1) (same shape)
    """
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation: exp(x) / sum(exp(x))
    Numerically stable version.
    
    Args:
        x: Input tensor (last dimension is class scores)
    
    Returns:
        Softmax probabilities (same shape, sums to 1 along last axis)
    """
    x = np.asarray(x, dtype=np.float64)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent: (exp(2x) - 1) / (exp(2x) + 1)
    
    Args:
        x: Input tensor
    
    Returns:
        Tanh output in (-1, 1) (same shape)
    """
    x = np.asarray(x, dtype=np.float64)
    return np.tanh(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU approximation).
    gelu(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    
    Args:
        x: Input tensor
    
    Returns:
        GELU output (same shape)
    """
    x = np.asarray(x, dtype=np.float64)
    cdf = 0.5 * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))
    return x * cdf


# ============= Optimizers =============

class AdamOptimizer:
    """Adam optimizer state and update rule."""
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = learning_rate
        self.beta1 = beta1  # Momentum
        self.beta2 = beta2  # RMSProp
        self.epsilon = 1e-8
        self.t = 0  # timestep
        self.m = {}  # first moment (momentum)
        self.v = {}  # second moment (RMSProp)
    
    def update(self, param_id: int, gradient: np.ndarray, param: np.ndarray) -> np.ndarray:
        """Update parameter using Adam rule."""
        self.t += 1
        
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(gradient)
            self.v[param_id] = np.zeros_like(gradient)
        
        # Update biased first moment estimate
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class SGDOptimizer:
    """Stochastic Gradient Descent with momentum."""
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, param_id: int, gradient: np.ndarray, param: np.ndarray) -> np.ndarray:
        """Update parameter using SGD with momentum."""
        if param_id not in self.velocity:
            self.velocity[param_id] = np.zeros_like(gradient)
        
        # Velocity update
        self.velocity[param_id] = (
            self.momentum * self.velocity[param_id] - self.lr * gradient
        )
        
        # Parameter update
        return param + self.velocity[param_id]


def adam(learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> Callable:
    """
    Create Adam optimizer.
    
    Args:
        learning_rate: Step size (Float[1])
        beta1: Exponential decay for momentum (Float[1])
        beta2: Exponential decay for RMSProp (Float[1])
    
    Returns:
        Optimizer function for use in autodiff blocks
    """
    optimizer = AdamOptimizer(learning_rate, beta1, beta2)
    return optimizer


def sgd(learning_rate: float = 0.01, momentum: float = 0.9) -> Callable:
    """
    Create SGD optimizer.
    
    Args:
        learning_rate: Step size (Float[1])
        momentum: Momentum coefficient (Float[1])
    
    Returns:
        Optimizer function for use in autodiff blocks
    """
    optimizer = SGDOptimizer(learning_rate, momentum)
    return optimizer


# ============= Layer Utilities =============

def linear(in_features: int, out_features: int) -> np.ndarray:
    """
    Initialize linear layer weights (HeNormal).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
    
    Returns:
        Weight matrix (out_features, in_features)
    """
    variance = 2.0 / in_features
    weights = np.random.normal(0, np.sqrt(variance), (out_features, in_features))
    return weights


def conv1d(in_channels: int, out_channels: int, kernel_size: int) -> np.ndarray:
    """
    Initialize 1D convolution kernel.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
    
    Returns:
        Kernel tensor (out_channels, in_channels, kernel_size)
    """
    variance = 2.0 / (in_channels * kernel_size)
    kernel = np.random.normal(0, np.sqrt(variance),
                             (out_channels, in_channels, kernel_size))
    return kernel


def batch_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Batch normalization: (x - mean) / sqrt(var + eps)
    
    Args:
        x: Input tensor
        eps: Small constant for numerical stability (Float[1])
    
    Returns:
        Normalized output (same shape)
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)


def dropout(x: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Dropout: randomly zero elements with probability p (during training).
    
    Args:
        x: Input tensor
        p: Dropout probability (Float[1] in [0, 1])
    
    Returns:
        Output with p fraction of elements zeroed, scaled by 1/(1-p)
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.random.binomial(1, 1 - p, x.shape)
    return x * mask / (1 - p + 1e-15)
