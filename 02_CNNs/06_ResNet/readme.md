# ResNet from Scratch

This notebook implements ResNet for image classification, following the [original paper](https://arxiv.org/pdf/1512.03385) as closely as possible.

## What is ResNet?

ResNet (Residual Network) introduced residual connections to solve the degradation problem in deep neural networks. Instead of learning a direct mapping, ResNet learns a residual function by adding shortcut connections that bypass one or more layers. This allows much deeper models to be trained more effectively.

## Common ResNet Variants

ResNet is available in several standard depths:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

In this notebook, the main implementation is based on the ResNet-18 architecture, using a basic residual block. The design can be extended to deeper variants by changing the number of blocks per layer and, for bottleneck versions, the block type.

## Implementation Notes

- The model is implemented from scratch in `ResNet_from_scratch.ipynb`.
- The structure is designed to match the original ResNet paper as closely as possible.
- The core components include:
  - 7x7 Conv + BatchNorm + ReLU + MaxPool
  - Residual blocks with two 3x3 convolutions
  - Shortcut connections for identity mapping
  - Adaptive average pooling and a final fully connected layer

## Training Setup

- Uses SGD with momentum as the optimizer.
- Includes weight decay for regularization.
- Uses learning rate scheduling and optional early stopping.
- Mixed precision training is enabled with `torch.amp`.

## Dataset

The notebook uses the Food101 dataset and splits the training set into train/validation subsets. The test set is used to evaluate final accuracy.

## Goals

The goal of this implementation is to show a clear and faithful ResNet architecture, demonstrating how the residual learning idea works in practice and how a ResNet model can be trained from scratch using PyTorch.