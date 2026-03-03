# LLMs and NNs Embedded from Scratch

**Status:** Ongoing research and development  
**Focus:** Large Language Models (LLMs), Neural Networks (NNs), and Transformer architectures for embedded systems

This repository contains a minimal, self-contained C++14 library for tensor operations and basic neural network layers.  
**Note:** This code is primarily for my own research, experimentation, and self-learning. It is not intended to compete with established libraries or for production use.

## Motivation

- **Deep Understanding:** Built from scratch to learn how tensor operations and neural network layers work under the hood, with a special focus on LLMs and transformer models.
- **Embedded/Custom Optimization:** Designed to be lightweight and modifiable for embedded or resource-constrained environments.
- **Full Control:** Direct memory management and simple data structures for maximum transparency.

## Features

- Custom `Tensor3D<T>` and `Tensor4D<T>` classes.
- Basic neural network layers (Dense, Conv).
- Simple activation functions (ReLU, Sigmoid, Softmax).
- C++14 compatibility.
- Foundation for experimenting with transformer and LLM components.


## Installation

Clone the repository and build with your preferred C++14 compiler (e.g., g++, clang, MSVC).


## Project Structure

- `Utilities/` - Core tensor and neural network utilities

## Contributing

Contributions are welcome for educational and research purposes. Please open an issue or pull request.

## License

MIT

## References

- [Eigen](https://eigen.tuxfamily.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

## Disclaimer

This project is for educational and personal research. Not intended for production use.
