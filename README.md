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
- GNU/Linux
## Usage

The application is often more advanced on the **`Dev` branch** than ** `main` branch**.

### Gaussian Filter — V1

```bash
./bin/Linux/64/Release/ImgProcLabDemo \
    --image resources/parrot.png \
    --kernel-size 7 \
    --version v1 \
    --algo gaussian \
    --sigma 1.5 \
    --benchmark \
    --visualize \
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


## Implemented Algorithms

| Algorithm | Version | Implementation | Status |
|-----------|---------|----------------|--------|
| Gaussian Filter | V1 | Naive 2D convolution | Implemented |
| Gaussian Filter | V2 | Separable convolution | Implemented |
| Gaussian Filter | V3 | Separable convolution with constant memory | Implemented |
| Gaussian Filter | V4 | Separable convolution with constant and shared memory | Not implemented yet |
| Median Filter | — | Naive 2D neighborhood filtering | Implemented (not refactored yet) |
| Average Filter | — | Naive 2D convolution | Implemented (not refactored yet) |
| Sobel Filter | — | Naive 2D convolution | Implemented (not refactored yet) |
| Grayscale Conversion | — | RGB to grayscale conversion | Implemented (not refactored yet) |
| Binary Thresholding | — | Pixel-wise thresholding | Implemented (not refactored yet) |

### Gaussian Filter — Implementation Configuration

Gaussian V3 and the planned V4 allow different convolution and memory-access strategies to be selected directly in the source code.

The configuration is defined through the `GaussianFilter` constructor in `AlgorithmRunner.cpp`.

**Convolution strategy:**

- `Separable::TRUE` — Separable convolution using two 1D passes.
- `Separable::FALSE` — Naive 2D convolution using a single 2D pass.

**Memory strategy:**

- `METHOD::CONSTANT` — Gaussian coefficients stored in CUDA constant memory.
- `METHOD::NAIVE` — Gaussian coefficients accessed without constant memory.

**Default V3 configuration:**

```cpp
ImgProcLab::GaussianVersion::V3,
Separable::TRUE,
METHOD::CONSTANT
```

To switch between implementations, simply change the corresponding enum values in the `GaussianFilter` constructor.

For example, to use naive 2D convolution without constant memory:

```cpp
ImgProcLab::GaussianVersion::V3,
Separable::FALSE,
METHOD::NAIVE
```

These options allow different convolution and memory-access strategies to be explored and benchmarked without changing the underlying CUDA kernels.

### Benchmark Configuration

By default, the automated benchmarking script (`run.sh`) executes the **optimized configuration of each implemented Gaussian version**:

- **V1:** Naive 2D convolution.
- **V2:** Separable convolution.
- **V3:** Separable convolution with constant memory.
- **V4 (planned):** Separable convolution with constant and shared memory.

Alternative configurations can be benchmarked by modifying the corresponding enum values in `AlgorithmRunner.cpp` and rebuilding the project.

**Note:** V4 is not yet implemented. Configuration changes currently require modifying the source code and rebuilding the application.

Additional CUDA optimizations and refactoring of existing implementations are planned.


## Benchmarking

ImgProcLab includes an automated benchmarking script for comparing CUDA implementations against OpenCV CPU and CUDA.

### Running the Benchmarks

From the repository root:

```bash
./run.sh
```

The script automatically cleans previous build artifacts, builds the project in Release mode, and benchmarks Gaussian implementations V1, V2, and V3 using kernel sizes from 3×3 to 31×31.

Each configuration runs 10 warmup iterations and 100 measured iterations, with a Gaussian sigma of 2.0.

**Note:** Update the `IMAGES` array in `run.sh` to reference other images available on your system.

## Development Status

The project currently prioritizes:

1. Implementing functional CUDA image-processing algorithms.
2. Exploring and comparing different GPU optimization strategies.
3. Validating results against reference implementations.
4. Benchmarking and profiling CUDA kernels.
5. Maintaining a clear and reusable C++ project structure.

Further documentation, benchmark results, and implementation details will be added as development progresses.
