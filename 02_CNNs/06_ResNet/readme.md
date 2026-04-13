# ResNet (Residual Networks)

## Overview
ResNet (Residual Network) is a framework for training very deep neural networks by making it easier for the optimizer to learn identity-like mappings. It was introduced to overcome practical difficulties when scaling depth in convolutional networks.

## The problem: degradation
In theory, a deeper network should perform at least as well as a shallower one because the extra layers could learn the identity mapping. In practice, deeper models often suffer from the degradation problem: training accuracy saturates and then degrades as depth increases. This is not caused by overfitting — the training error itself increases for very deep plain networks.

## The solution: residual learning
Instead of forcing a stack of layers to directly approximate the desired mapping $\mathcal{H}(x)$, ResNet lets these layers learn a residual mapping $\mathcal{F}(x)$, defined as:

$$\mathcal{F}(x) := \mathcal{H}(x) - x$$

Thus the original mapping becomes:

$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

The intuition is that it is easier for the optimizer to push the residual $\mathcal{F}(x)$ toward zero (i.e., learn small corrections) than to learn an identity mapping through many nonlinear layers.

## Residual block
The core building block of ResNet is the residual block, which uses shortcut (skip) connections:
- Identity shortcut: the input $x$ is added directly to the output of the stacked layers: $y = \mathcal{F}(x, \{W_i\}) + x$.
- No extra parameters: identity shortcuts add negligible computation and no parameters.
- Dimension matching: if the dimensions of $x$ and $\mathcal{F}$ differ (e.g., different channel counts or spatial resolution), a linear projection $W_s$ can be applied on the shortcut:
  $y = \mathcal{F}(x, \{W_i\}) + W_sx$

## Architectures
The original ResNet paper compares plain (stacked) networks with residual networks and demonstrates that residual connections enable much deeper models to be trained reliably.

### Bottleneck design
For very deep variants (ResNet-50, -101, -152), a bottleneck block reduces computational cost. It consists of three layers:
1. $1\times1$ convolution to reduce dimensionality.
2. $3\times3$ convolution (the bottleneck).
3. $1\times1$ convolution to restore dimensionality.

This design reduces the number of parameters and operations while preserving representational capacity.

## Practical notes
- Residual connections make optimization easier and improve gradient flow.
- Use projection shortcuts when changing feature map sizes (e.g., during downsampling).
- Batch normalization and ReLU activations are commonly used inside residual blocks.

## Reference
K. He, X. Zhang, S. Ren, J. Sun. "Deep Residual Learning for Image Recognition" (2015).
