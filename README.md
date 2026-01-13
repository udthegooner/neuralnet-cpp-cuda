# CUDA-Accelerated Deep Learning Engine: MNIST

A high-performance Neural Network training and inference engine built from scratch in C++ and CUDA. This project demonstrates the implementation of custom CUDA kernels for deep learning primitives, optimized memory access patterns (coalescing), and a comparative performance analysis between CPU and GPU architectures.

## 🚀 Performance Benchmarks
The following results were obtained training on the MNIST dataset (60,000 training images, 10,000 test images) for **15 epochs** with a **batch size of 64**.

### **Hardware Acceleration Results**
| Metric | CPU Implementation | GPU (NVIDIA MX450) | Speedup |
| :--- | :--- | :--- | :--- |
| **Training Time** | 383,225 ms (~6.4 min) | **22,240.5 ms (~22.2 sec)** | **17.2x** |
| **Test Accuracy** | 90.87% | **97.37%** | **+6.5% Absolute** |
| **Inference Time (10k)** | 1,036.35 ms | **24.34 ms** | **42.6x** |

> **Key takeaway:** The GPU implementation achieved significantly higher accuracy and faster convergence by utilizing **LogSoftmax** for numerical stability and a custom **RMSProp** ($\beta=0.99$) optimization kernel.

---

## 🛠 Model Configuration
The engine executes a deep fully-connected architecture:
* **Architecture:** 784 (Input) → 128 (ReLU) → 64 (ReLU) → 10 (Output)
* **Activations:** ReLU, LogSoftmax (for numerical stability)
* **Optimization:** RMSProp with $\beta=0.99$ (GPU) / SGD (CPU)
* **Loss Function:** Negative Log-Likelihood (NLL) Loss

---

## 💡 Engineering Optimizations

### **1. Memory Coalescing & Throughput**
Standard matrix operations often suffer from strided memory access. This engine implements custom transpose kernels to ensure that threads within a warp access adjacent memory addresses. This groups memory requests into single 128-byte transactions, saturating the MX450's memory bandwidth.



### **2. Numerical Stability (Log-Space)**
To prevent gradient overflow/underflow, the output layer utilizes `LogSoftmax`. By working in log-probability space, the engine maintains high precision during backpropagation, contributing to the superior 97.37% accuracy.

### **3. Thrust Parallel Primitives**
Integrated the **NVIDIA Thrust** library for high-level data management and parallel reduction operations, specifically for calculating model accuracy (`thrust::inner_product`) and managing device-side memory vectors.

---

## 💻 Building and Running

### **Prerequisites**
* **CUDA Toolkit:** 11.0 or higher
* **Build System:** CMake 3.18+
* **Compiler:** MSVC (Visual Studio 2022) or GCC 9+
* **GPU:** NVIDIA GeForce MX450 or any CUDA-enabled GPU (Pascal+ architecture)

### **Compilation**
The build process is managed via CMake. Toggle the `USE_GPU` flag to switch between hardware targets.

```bash
# 1. Create and enter build directory
mkdir build && cd build

# 2. Configure for GPU (Set to OFF for CPU-only build)
cmake .. -DUSE_GPU=ON

# 3. Build the Release binary
cmake --build . --config Release