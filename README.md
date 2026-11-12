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
- CUDA Toolkit 13.4
- OpenCV 4 (for image loading and reference implementations)
- CMake 3.18 or later
- C++17-compatible compiler
- GNU/Linux
## Usage

The application is often more advanced on the **`Dev` branch** than the **`main` branch**.

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
| `--benchmark` | Benchmark mode |
| `--visualize` | Visualization mode |

Additional algorithms and CUDA implementations are planned and will be integrated progressively.


## Implemented Algorithms

| Algorithm | Version | Implementation | Status |
|-----------|---------|----------------|--------|
| Gaussian Filter | V1 | Naive 2D convolution | Implemented |
| Gaussian Filter | V2 | Separable convolution | Implemented |
| Gaussian Filter | V3 | Separable convolution with constant memory | Implemented |
| Gaussian Filter | V4 | Separable convolution with shared memory | Implemented |
| Gaussian Filter | V5 | Separable convolution with constant memory and shared memory | Implemented |
| Median Filter | — | Naive 2D neighborhood filtering | Implemented (not refactored yet) |
| Average Filter | — | Naive 2D convolution | Implemented (not refactored yet) |
| Sobel Filter | — | Naive 2D convolution | Implemented (not refactored yet) |
| Grayscale Conversion | — | RGB to grayscale conversion | Implemented (not refactored yet) |
| Binary Thresholding | — | Pixel-wise thresholding | Implemented (not refactored yet) |

### Gaussian Filter — Implementation Configuration

Gaussian **V3**, **V4** and **V5** allow different convolution and memory-access strategies to be selected directly in the source code.

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
- **V4:** Separable convolution with shared memory.
- **V5:** Separable convolution with constant and shared memory.

Alternative configurations can be benchmarked by modifying the corresponding enum values in `AlgorithmRunner.cpp` and rebuilding the project.

**Note:** Configuration changes currently require modifying the source code and rebuilding the application.

Additional CUDA optimizations and refactoring of existing implementations are planned.


## Benchmarking

ImgProcLab includes an automated benchmarking script for comparing CUDA implementations against OpenCV CPU and CUDA.

### Running the Benchmarks

From the repository root:

```bash
./run.sh
```

The script automatically cleans previous build artifacts, builds the project in Release mode, and benchmarks Gaussian implementations **V1**, **V2**, **V3**, **V4** and **V5**
using kernel sizes from **3×3** to **31×31**.

Each configuration runs 10 warmup iterations and 100 measured iterations, with a Gaussian sigma of 1.5.

**Note:** Update the `IMAGES` array in `run.sh` to reference other images available on your system.



## Results 
The images used can be found in the `resources` directory.

### Correctness

#### Filter Correctness: Mean Absolute Error (MAE) and Maximum Absolute Error (MAE) 

Before presenting the results, it is important to describe the approach used to evaluate the correctness of the implementations.
The output of each implemented version is compared pixel by pixel against both OpenCV's CPU and CUDA implementations.

Two metrics are used for this comparison: **Mean Absolute Error (MAE)** and **Maximum Absolute Error**.

- **Mean Absolute Error (MAE):** Measures the average absolute difference between corresponding pixel values across all image channels.
- **Maximum Absolute Error:** Measures the largest absolute difference between corresponding pixel values across all image channels.


##### Filter Correctness Results

The following results were obtained using the same configuration for all five versions:

- **Image resolution:** 1920×1080
- **Gaussian kernel size:** 5
- **Sigma:** 2.5

**Mean Absolute Error (MAE)**

| Version | CPU vs CUDA | ImgProcLab vs CPU | ImgProcLab vs CUDA |
|---------|-------------|-------------------|--------------------|
| V1      | 0.00729897  | 0.00729961        | 8.08427e-06        |
| V2      | 0.00729897  | 0.00729864        | 9.70113e-07        |
| V3      | 0.00729897  | 0.00729864        | 9.70113e-07        |
| V4      | 0.00729897  | 0.00729864        | 9.70113e-07        |
| V5      | 0.00729897  | 0.00729864        | 9.70113e-07        |

**Maximum Absolute Error**

| Version | CPU vs CUDA | ImgProcLab vs CPU | ImgProcLab vs CUDA |
|---------|-------------|-------------------|--------------------|
| V1      |      1      |        1          |         1          |
| V2      |      1      |        1          |         1          |
| V3      |      1      |        1          |         1          |
| V4      |      1      |        1          |         1          |
| V5      |      1      |        1          |         1          |

##### Observations

As shown in the tables, the maximum absolute difference between corresponding pixel values is **1** for all implementations.
Furthermore, the Mean Absolute Error is very close to zero, indicating that the outputs of all five versions are nearly identical to those produced by OpenCV's CPU and CUDA implementations.
The results obtained from V1 through **V5** demonstrate a very high level of numerical agreement with OpenCV's implementations.


### Gaussian V1: Naive Approach 

<table>
  <tr>
    <td><img src="reports/graphs/V1.0.png" width="600"></td>
    <td><img src="reports/graphs/V1.1.png" width="600"></td>
  </tr>
  <tr>
    <td><img src="reports/graphs/V1.2.png" width="600"></td>
    <td><img src="reports/graphs/V1.3.png" width="600"></td>
  </tr>
  <tr> 
    <td><img src="reports/graphs/V1.4.png" width="600"></td>
    <td><img src="reports/graphs/V1.5.png" width="600"></td>
  </tr>
</table>



The images above show the results obtained using a Gaussian filter on five images of different sizes, with 15 different kernel sizes and a sigma value of 1.5.

The approach used is a non-optimized (naive) implementation. The filtering operation, i.e., smoothing, is applied in both image dimensions within the same nested loop, resulting in an algorithm 
with O(n²) complexity.

This **O(n²)** behavior can be observed in the graphs, where the trend becomes more pronounced as the kernel size increases.

Bellow the graph comparing, the execution time and average computation time.


</table> 

 **Average Execution Time per Image (ms)**

| Image Size | CPU (ms) | CUDA (ms) | V1 (ms) |
|------------|----------|-----------|---------|
| 453×340    | 0.262416 | 0.020690  | 0.154490 |
| 532×532    | 0.424069 | 0.033154  | 0.255565 |
| 597×335    | 0.325389 | 0.024617  | 0.192899 |
| 540×720    | 0.492039 | 0.042909  | 0.342717 |
| 1920×1080  | 1.708543 | 0.184896  | 1.587830 |

 **Overall Average Execution Time (ms)**

| Implementation | Average Time (ms) |
|----------------|-------------------|
| CPU            | 0.642491          |
| CUDA           | 0.061253          |
| V1             | 0.506700          |


The results obtained show that **V1** is approximately **1.27×** faster on average than OpenCV's CPU 
Gaussian filter implementation, but **8.27×** slower on average than its CUDA implementation. 
Despite being a naive approach, one can observe the performance improvement achieved simply 
through parallelization.



### Gaussian V2: Separable filters 

<table>
  <tr>
    <td><img src="reports/graphs/V2.0.png" width="600"></td>
    <td><img src="reports/graphs/V2.1.png" width="600"></td>
  </tr>
  <tr>
    <td><img src="reports/graphs/V2.2.png" width="600"></td>
    <td><img src="reports/graphs/V2.3.png" width="600"></td>
  </tr>
  <tr> 
    <td><img src="reports/graphs/V2.4.png" width="600"></td>
    <td><img src="reports/graphs/V2.5.png" width="600"></td>
  </tr>
</table>


The result above was obtained by applying separable filters. A separable filter is a filter that can be decomposed into two or more simpler filters. 
In the case of the separable Gaussian filter, the same result can be obtained by applying a vertical filter followed by a horizontal filter, or vice versa.

As we can see, **V2** not only has better performance than V1 but also the CUDA implementation of OpenCV with the selected images. 
Unlike **V1**, **V2** has a computational complexity of **O(n)**, where n is the kernel size. 
All subsequent versions of the Gaussian filter will use filter separability as their foundation.


</table> 

**Average execution time per image (ms)**

| Image resolution | CPU      | CUDA     | V2       |
|------------------|----------|----------|----------|
| 453×340          | 0.287248 | 0.020497 | 0.016993 |
| 532×532          | 0.505921 | 0.033513 | 0.027844 |
| 597×335          | 0.335010 | 0.024888 | 0.020815 |
| 540×720          | 0.509607 | 0.042391 | 0.036611 |
| 1920×1080        | 1.708984 | 0.193626 | 0.174197 |

**Overall average execution time (ms)**

| Version | Average time (ms) |
|---------|-------------------|
| CPU     | 0.669354          |
| CUDA    | 0.062983          |
| V2      | 0.055292          |


The results obtained, show that **V2** achieves an average execution time **91.74% lower** than OpenCV's CPU Gaussian filter implementation 
and **12.21% lower** than its CUDA implementation. 
These results demonstrate the performance benefits of GPU parallelization and the effectiveness of separable convolution.


### Gaussian V3: Separable filters with constant memory

<table>
  <tr>
    <td><img src="reports/graphs/V3.0.png" width="600"></td>
    <td><img src="reports/graphs/V3.1.png" width="600"></td>
  </tr>
  <tr>
    <td><img src="reports/graphs/V3.2.png" width="600"></td>
    <td><img src="reports/graphs/V3.3.png" width="600"></td>
  </tr>
  <tr> 
    <td><img src="reports/graphs/V3.4.png" width="600"></td>
    <td><img src="reports/graphs/V3.5.png" width="600"></td>
  </tr>
</table>


The result above was obtained by applying separable filters with constant memory.
An obvious optimization arises from the fact that the filter coefficients remain unchanged throughout execution.
In the previous approaches, **V1** and **V2**, the filter coefficients are accessed from global memory, which generally has high latency.
Constant memory is a read-only memory space with a dedicated cache, making it particularly efficient when threads within a warp access the same address, as is the case in our implementation.


**Average execution time per image (ms)**

| Image resolution | CPU      | CUDA     | V3       |
|------------------|----------|----------|----------|
| 453×340          | 0.280530 | 0.020500 | 0.015732 |
| 532×532          | 0.469630 | 0.033939 | 0.024597 |
| 597×335          | 0.327891 | 0.025094 | 0.018275 |
| 540×720          | 0.499284 | 0.043300 | 0.031499 |
| 1920×1080        | 1.736945 | 0.195336 | 0.149522 |

**Overall average execution time (ms)**

| Version | Average time (ms) |
|---------|-------------------|
| CPU     | 0.662856          |
| CUDA    | 0.063634          |
| V3      | 0.047925          |


The results obtained show that **V3** reduces the average execution time by approximately **92.76%** compared to OpenCV's CPU Gaussian filter implementation, 
which represents a considerable improvement.
Furthermore, **V3** reduces the average execution time by approximately **24.68%** compared to OpenCV's CUDA implementation.



### Gaussian V4: Separable filters with shared memory

<table>
  <tr>
    <td><img src="reports/graphs/V4.0.png" width="600"></td>
    <td><img src="reports/graphs/V4.1.png" width="600"></td>
  </tr>
  <tr>
    <td><img src="reports/graphs/V4.2.png" width="600"></td>
    <td><img src="reports/graphs/V4.3.png" width="600"></td>
  </tr>
  <tr> 
    <td><img src="reports/graphs/V4.4.png" width="600"></td>
    <td><img src="reports/graphs/V4.5.png" width="600"></td>
  </tr>
</table>

The result above was obtained by applying separable filters with shared memory.
Shared memory is designed as part of the CUDA device SM (**Streaming Multiprocessor**), which can be seen as the equivalent of CPU core.
The access of data in shared memory is similar to accessing data to global memory, but since the shared memory resides on the chip, it has a lower latency and higher throughput 
than the global memory. 

Furthermore, the OpenCV CUDA implementation uses the shared memory approach for the gaussian filter with some other optimizations such multiple output pixels per thread which **V4** 
does not include. 
The result above shows us that **V4** is competitive with OpenCV CUDA implementation.

For more details on OpenCV's CUDA implementation, refer to the CUDA column filter source [code](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudafilters/src/cuda/column_filter.hpp).


**Average execution time per image (ms)**

| Image resolution | CPU      | CUDA     | V4       |
|------------------|----------|----------|----------|
| 453×340          | 0.326418 | 0.021712 | 0.022046 |
| 532×532          | 0.441967 | 0.033617 | 0.034727 |
| 597×335          | 0.346635 | 0.024932 | 0.025859 |
| 540×720          | 0.495365 | 0.042628 | 0.046839 |
| 1920×1080        | 1.687100 | 0.196126 | 0.225980 |

**Overall average execution time (ms)**

| Version | Average time (ms) |
|---------|-------------------|
| CPU     | 0.659497          |
| CUDA    | 0.063803          |
| V4      | 0.071090          |


The results obtained show that **V4** reduces the average execution time by approximately **89.22%** compared to OpenCV's CPU implementation, 
but increases the average execution time by approximately **11.42%** compared to OpenCV's CUDA Gaussian filter implementation.


### Gaussian V5: Separable filters with constant memory and shared memory

<table>
  <tr>
    <td><img src="reports/graphs/V5.0.png" width="600"></td>
    <td><img src="reports/graphs/V5.1.png" width="600"></td>
  </tr>
  <tr>
    <td><img src="reports/graphs/V5.2.png" width="600"></td>
    <td><img src="reports/graphs/V5.3.png" width="600"></td>
  </tr>
  <tr> 
    <td><img src="reports/graphs/V5.4.png" width="600"></td>
    <td><img src="reports/graphs/V5.5.png" width="600"></td>
  </tr>
</table>


The result above was obtained by applying separable filters with shared memory and constant memory.
The overall result is better than the OpenCV CUDA implementation but not better than **V3**.


**Average execution time per image (ms)**

| Image resolution | CPU      | CUDA     | V5       |
|------------------|----------|----------|----------|
| 453×340          | 0.287598 | 0.020575 | 0.019065 |
| 532×532          | 0.444237 | 0.033968 | 0.029530 |
| 597×335          | 0.337051 | 0.024624 | 0.022027 |
| 540×720          | 0.495823 | 0.042546 | 0.038996 |
| 1920×1080        | 1.674898 | 0.198119 | 0.186311 |

**Overall average execution time (ms)**

| Version | Average time (ms) |
|---------|-------------------|
| CPU     | 0.647921          |
| CUDA    | 0.063966          |
| V5      | 0.059186          |


The results obtained show that **V5** reduces the average execution time by approximately **90.86%** compared to OpenCV's CPU implementation, 
but reduces the average execution time by approximately **7.47%** compared to OpenCV's CUDA implementation.


### Gaussian Filter Conclusion 

In this section, the results obtained using different optimization approaches were presented and compared. The results illustrate the power of parallelization, particularly with **V1**. 
However, parallelization alone is not enough to achieve optimal performance. **V2** improved performance by introducing separable filters, while V3 incorporated constant memory to further reduce 
memory access latency. **V3** proved to be the best-performing approach overall, achieving better results than **V5** despite using fewer optimization techniques.

I believe that, with further optimizations, all versions from **V2** to **V5** could achieve even better performance than OpenCV's CUDA implementation of the Gaussian filter.
Using Nsight Compute, memory latency and occupancy were identified as major performance concerns, particularly in **V5**.


## Development Status

The project currently prioritizes:

1. Implementing functional CUDA image-processing algorithms.
2. Exploring and comparing different GPU optimization strategies.
3. Validating results against reference implementations.
4. Benchmarking and profiling CUDA kernels.
5. Maintaining a clear and reusable C++ project structure.

Further documentation, benchmark results, and implementation details will be added as development progresses.

## Personal Notes (TODO)

1. Automated testing with CTest.
2. Robust CUDA error handling and Compute Sanitizer checks.
3. Clean CMake configuration and reproducible builds.
4. Clear internal C++ architecture and resource ownership.
5. GitHub Actions CI and polished documentation.

