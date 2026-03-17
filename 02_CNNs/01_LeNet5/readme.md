# LeNet-5 Implementation on MNIST

This repository contains a PyTorch implementation of the classic **LeNet-5** architecture, trained on the MNIST dataset of handwritten digits.

### 1. What is LeNet-5?
LeNet-5 is a pioneering Convolutional Neural Network (CNN) architecture designed by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner in their 1998 paper. It was specifically created to solve the problem of handwritten digit recognition (the MNIST dataset). It introduced fundamental concepts that are still used in modern computer vision, such as **convolutional layers**, **subsampling (pooling)**, and **fully connected layers**.

### 2. Is this script exactly like the paper?
The implementation in `LeNet5.ipynb` is a **modernized version** of the original 1998 architecture. While it retains the core sequential structure, it uses modern deep learning practices for efficiency:

| Feature | Original Paper (1998) | This Implementation (`LeNet5.ipynb`) |
| :--- | :--- | :--- |
| **Connectivity** | C3 was partially connected to S2 to save parameters. | **Dense connection**: All 6 maps connect to all 16 maps. |
| **Subsampling** | Trainable weights/biases after average pooling. | **Fixed Average Pooling** (`nn.AvgPool2d`). |
| **Activations** | Scaled Tanh: $1.7159 \cdot \tanh(\frac{2}{3}a)$. | Standard `nn.Tanh()`. |
| **Output Layer** | Radial Basis Function (RBF) units. | **Linear Layer** with Softmax/CrossEntropy. |
| **Training** | Complex second-order optimization. | **SGD (Stochastic Gradient Descent)**. |
