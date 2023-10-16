# kmeans-CUDA
# KMeans with CUDA & Thrust

[![Build Status](https://travis-ci.com/your_username/kmeans-cuda-thrust.svg?branch=main)](https://travis-ci.com/your_username/kmeans-cuda-thrust)

This repository contains an efficient implementation of the KMeans clustering algorithm using NVIDIA CUDA and Thrust. The implementation aims to leverage the parallelism capabilities of GPUs to speed up the clustering process.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Features

- Efficient GPU-based KMeans clustering.
- Uses Thrust library for easier CUDA algorithm development.
- Scalable for large datasets.
- Offers significant performance gains over CPU implementations.

## Prerequisites

- NVIDIA GPU (Compute Capability 3.5 or higher recommended)
- CUDA Toolkit (10.0 or higher recommended)
- CMake (3.8 or higher)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/kmeans-cuda-thrust.git
cd kmeans-cuda-thrust
```
Create a build directory and compile:
```bash
mkdir build
cd build
make
```
## Usage
After building, you can run the KMeans algorithm as follows:
```bash
./bin/kmeans -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -c -s 8675309 
```

## Benchmarks
| Implementation          | Iterations | Time per Iteration (ms) | Speedup compared to CPU |
|-------------------------|------------|-------------------------|-------------------------|
| Baseline (CPU)          | 14         | 42.111634               | 1x                      |
| Basic CUDA              | 34         | 6.946278                | 6.06x                   |
| CUDA with Shared Memory | 48         | 4.558344                | 9.24x                   |
| Thrust                  | 34         | 14.937456               | 2.82x                   |


## Contributing
Fork the repository.
Create your feature branch (git checkout -b feature/my-new-feature).
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/my-new-feature).
Create a new Pull Request.
