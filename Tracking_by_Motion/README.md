# Object Tracking by Motion

This repository focuses on Object Tracking by Motion (detector-free tracking). The tracking algorithms can be categorized into the following approaches based on their underlying techniques:

## Optical Flow
The classic approach. Calculates a motion vector between two frames for every pixel (dense) or selected points (sparse).
* **Lucas-Kanade (1981)**: Sparse and fast.
* **Horn-Schunck**: Dense and global.
* **Farnebäck**: The current OpenCV standard for dense flow.

## Feature Trackers
Track prominent points across frames.
* **KLT (Kanade-Lucas-Tomasi)**: Detects corners using Shi-Tomasi and tracks them with optical flow.
* **RAFT and FlowNet/PWC-Net**: Replace classic mathematical models with CNNs or GRUs – significantly more robust against occlusions and large movements.

## Background Subtraction
Separates moving foreground regions from the static background.
* **MOG2**: Models each pixel as a Gaussian Mixture and is robust against shadows.
* **ViBe**: Maintains a sample of past pixel values and is extremely fast, making it ideal for embedded applications.

## Density-based Trackers
No explicit detector is used; instead, a color distribution is maximized.
* **Mean-Shift**: Iteratively searches for the mode of the color density.
* **CamShift**: Extends Mean-Shift with adaptive window sizes.
* **Particle Filter (CONDENSATION)**: Scatters hypotheses in the state space – ideal for non-linear motion and occlusions.

## Modern / Deep Learning-based (Detector-free)
The latest generation that works entirely without a traditional bounding box detector.
* **CoTracker (Meta AI) & TAPIR (DeepMind)**: Track arbitrary points across long videos using Transformer architectures.
* **DINO Tracker**: Uses self-supervised ViT features for zero-shot capable tracking.

## Segmentation-based
Propagates masks instead of boxes.
* **SAM 2 (Meta, 2024)**: Currently the most powerful system in this category. A single prompt (click, box, or mask) in the first frame is enough, and the mask is propagated throughout the entire video – completely eliminating the need for a classic detector.