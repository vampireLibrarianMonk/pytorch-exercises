# Purpose
A baseline repo to practice a series of increasingly difficult PyTorch exercises.

# PyTorch
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It is widely used for applications in deep learning, a subset of machine learning where neural networks—models inspired by the human brain—learn from large amounts of data. PyTorch is known for its ease of use, efficiency, and dynamic computational graph that allows flexibility in machine learning model design. It is predominantly used with Python and is popular in both academic research and industry applications. PyTorch's intuitive design and strong community support make it suitable for both beginners and advanced users. It is used in various fields ranging from computer vision to natural language processing.

# Torch Vision
Torchvision is a part of the PyTorch project and is specifically designed for computer vision applications. It offers a collection of popular datasets, model architectures, and common image transformations. The package is classified into stable, beta, and prototype features based on their development and stability status. Stable features are long-term supported with no major performance limitations, beta features are subject to change based on user feedback, and prototype features are in early stages for testing. Torchvision facilitates transforming and augmenting images, provides pre-trained weights for various models, and supports diverse datasets and utilities for tasks like object detection and semantic segmentation.

# Supported GPUs
PyTorch's GPU support includes a variety of hardware:

  * NVIDIA: PyTorch fully supports NVIDIA GPUs via the CUDA toolkit, which provides highly optimized performance for deep learning models and large datasets.

  * Intel: While PyTorch can run on Intel CPUs, GPU support via Intel's hardware (like the oneAPI Deep Neural Network Library or oneDNN) is less optimized compared to NVIDIA.

  * AMD: PyTorch support for AMD GPUs is growing, particularly through the ROCm (Radeon Open Compute) platform, though it might not be as extensive as NVIDIA's CUDA support.

  * Mac GPUs: PyTorch supports GPU acceleration on Mac systems with Apple Silicon (such as the M1 and later models) through the "Metal Performance Shaders (MPS) backend for PyTorch." This integration allows efficient utilization of Apple's Metal technology, enhancing performance over CPU-only execution. Users need a compatible PyTorch version and systems meeting Metal Performance Shaders requirements, with details available on PyTorch's official website or GitHub repository.

**When working with PyTorch, it's essential to consider the compatibility and performance optimization with the specific hardware and software stack being used.**

# Repositories
## Pytorch
  * [GitHub](https://github.com/pytorch/pytorch)
  * [Anaconda](https://anaconda.org/pytorch/pytorch)
  * [PyPI](https://pypi.org/project/torch/)

## TorchVision
  * [Github](https://github.com/pytorch/vision)
  * [Anaconda](https://anaconda.org/pytorch/torchvision)
  * [Pypi](https://pypi.org/project/torchvision/)

# Courses:
  * [Udemy: A deep understanding of deep learning (with Python intro)](https://www.udemy.com/course/deeplearning_x/)
  * [ZTM: PyTorch for Deep Learning in 2024](https://zerotomastery.io/courses/learn-pytorch/)

# Useful Links:
  * [PyTorch Tutorials](https://pytorch.org/)
  * [PyTorch Examples Repository](https://github.com/pytorch/examples)
  * [PyTorch Discussion Forums](https://discuss.pytorch.org/)
  * [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  * [PyTorch – Visualization of Convolutional Neural Networks in PyTorch](https://datahacker.rs/028-visualization-and-understanding-of-convolutional-neural-networks-in-pytorch/)
