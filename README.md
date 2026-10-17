# ImgProcLab

**An Image Processing Library in CUDA C++**

## First Things First

This repository is under active development and is currently in the **prototyping phase**.

The primary goal at this stage is to establish a functional, maintainable, and reasonably professional codebase without unnecessary overengineering.

While certain architectural decisions and good software engineering practices are followed from the beginning, the current focus is on implementing, testing, and benchmarking image-processing algorithms and CUDA optimizations.

Code refinement, additional robustness checks, and further improvements to the overall design will be addressed incrementally as the project matures.

## Scope of This Repository

As Richard P. Feynman once wrote:

> *"What I cannot create, I do not understand."*

This repository reflects that philosophy through the implementation of image-processing algorithms using CUDA C++.

Rather than studying concepts purely in theory, my goal is to develop a deeper understanding of how they work by translating them into functioning code, identifying gaps in my knowledge, and addressing them through experimentation.

Image processing provides a practical foundation for this exploration. It offers both an opportunity to revisit fundamental concepts and a chance to implement algorithms that I have frequently used through high-level APIs.

By implementing these algorithms from the ground up, I aim to gain a better understanding of their underlying computations, memory-access patterns, parallelization strategies, and GPU-specific performance considerations.

The project also serves as a platform for experimenting with CUDA optimization techniques and comparing different implementations against established libraries such as OpenCV.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4 (for image loading and reference implementations)
- CMake 3.18 or later
- C++17-compatible compiler

## Usage

The application is currently available on the **`Dev` branch**.

### Gaussian Filter — V1

```bash
./ImgProcLabDemo \
    --image /path/to/image.png \
    --kernel-size 7 \
    --version v1 \
    --algo gaussian \
    --sigma 1.5
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--image` | Path to the input image |
| `--algo` | Image-processing algorithm to execute |
| `--version` | CUDA implementation version |
| `--kernel-size` | Kernel size (positive odd integer) |
| `--sigma` | Gaussian standard deviation |

Additional algorithms and CUDA implementations are planned and will be integrated progressively.

## Development Status

The project currently prioritizes:

1. Implementing functional CUDA image-processing algorithms.
2. Exploring and comparing different GPU optimization strategies.
3. Validating results against reference implementations.
4. Benchmarking and profiling CUDA kernels.
5. Maintaining a clear and reusable C++ project structure.

Further documentation, benchmark results, and implementation details will be added as development progresses.