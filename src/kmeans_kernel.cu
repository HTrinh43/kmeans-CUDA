#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>
#include "helpers.h"
#include <cmath>
#include <chrono>

__global__ void assignLabels(float *points, float *centroids, int *labels, int n_points, int k, int n_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    int nearestCentroid = 0;
    float minDist = FLT_MAX;

    for (int centroidIdx = 0; centroidIdx < k; centroidIdx++) {
        float dist = 0.0f;

        for (int dim = 0; dim < n_dim; dim++) {
            float diff = points[idx * n_dim + dim] - centroids[centroidIdx * n_dim + dim];
            dist += diff * diff;
        }

        if (dist < minDist) {
            minDist = dist;
            nearestCentroid = centroidIdx;
        }
    }

    labels[idx] = nearestCentroid;
}

void findClosestCentroid(float* points, int* d_labels, int n_points, int n_dim, int k,float* d_centroids) {
    int blockSize = 256;
    int gridSize = (n_points + blockSize - 1) / blockSize;

    // Launching the kernel
    assignLabels<<<gridSize, blockSize>>>(points, d_centroids, d_labels, n_points, k, n_dim);

    // Synchronize to ensure the kernel has completed
    cudaDeviceSynchronize();

}


__global__ void checkConvergenceKernel(float* d_centroids, float* d_old_centroids, float* d_diffs, int k, int n_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * n_dim) {
        float diff = d_centroids[idx] - d_old_centroids[idx];
        d_diffs[idx] = diff * diff;  // store squared difference
    }
}

bool hasConverged(float* d_centroids, float* d_old_centroids, float* d_diffs, int k, int n_dim, float threshold) {
    // Compute squared differences using the kernel
    checkConvergenceKernel<<<(k * n_dim + 255) / 256, 256>>>(d_centroids, d_old_centroids, d_diffs, k, n_dim);
    cudaDeviceSynchronize(); // Ensure kernel completion

    // Copy the squared differences to host
    float* h_diffs = new float[k * n_dim];
    cudaMemcpy(h_diffs, d_diffs, k * n_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute total difference on the host
    float totalDiff = 0.0f;
    for (int i = 0; i < k * n_dim; i++) {
        totalDiff += h_diffs[i];
    }

    delete[] h_diffs; // Clean up

    // Check for convergence and return the result

    return sqrt(totalDiff) < threshold;
}



__global__ void updateCentroidsKernel(float* points, int* labels, float* d_centroids_sums, int* d_counts, int n_points, int n_dim, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_points) {
        for (int d = 0; d < n_dim; d++) {
            atomicAdd(&d_centroids_sums[labels[idx] * n_dim + d], points[idx * n_dim + d]);
        }
        atomicAdd(&d_counts[labels[idx]], 1);
    }
}

__global__ void computeNewCentroidsKernel(float* d_centroids_sums, int* d_counts, float* d_centroids, int n_dim, int k) {
    int centroid_idx = blockIdx.x;
    int dim = threadIdx.x;

    if (d_counts[centroid_idx] > 0) {  // Avoid division by zero
        d_centroids[centroid_idx * n_dim + dim] = d_centroids_sums[centroid_idx * n_dim + dim] / d_counts[centroid_idx];
    }
    else {
        d_centroids[centroid_idx * n_dim + dim] = 0;
    }
}

void updateCentroids(float* d_points,int* d_labels, float* d_centroids, int n_points, int n_dim, int k, 
float* d_centroids_sums, int* d_counts) {
    
    cudaMemset(d_centroids_sums, 0, k * n_dim * sizeof(float));
    cudaMemset(d_counts, 0, k * sizeof(int));
    int blockSize = 256;  // Or whatever number is optimal for your GPU and problem.
    int gridSize = (n_points + blockSize - 1) / blockSize;


    // Sum up the points for each centroid
    updateCentroidsKernel<<<gridSize, blockSize>>>(d_points, d_labels, d_centroids_sums, d_counts, n_points, n_dim, k);
    cudaDeviceSynchronize();

    // Compute new centroids
    computeNewCentroidsKernel<<<k, n_dim>>>(d_centroids_sums, d_counts, d_centroids, n_dim, k);
    cudaDeviceSynchronize();

    // cudaFree(d_centroids_sums);
    // cudaFree(d_counts);
}


float* kmeansCUDA(Point_cu data, kmeans_args_t args) {


    // Create CUDA events for timing
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, 0);

    // Dynamic allocation of device-side variables
    int k = args.k;
    int n_dim = args.d;
    int n_val = data.size;
    int max_iters = args.m;
    float threshold = args.t;
    int seed = args.s;

    float* d_points;
    float* d_centroids;
    float* d_old_centroids;
    int* d_labels;
    cudaMalloc(&d_points, n_val * n_dim * sizeof(float));
    cudaMalloc(&d_centroids, k * n_dim * sizeof(float));
    cudaMalloc(&d_old_centroids, k * n_dim * sizeof(float));
    cudaMalloc(&d_labels, n_val * sizeof(int));

    float* d_centroids_sums;
    int* d_counts;

    cudaMalloc(&d_centroids_sums, k * n_dim * sizeof(float));
    cudaMalloc(&d_counts, k * sizeof(int));
    // Initialize centroids on host
    float* h_centroids = initializeCentroids(data, n_val,n_dim, k, seed);
    float* d_diffs;

    auto start1 = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&d_diffs, k * n_dim * sizeof(float));

    // Transfer centroids to device
    cudaMemcpy(d_centroids, h_centroids, k * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, data.features, n_val * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    // const int threadsPerBlock = 128;
    // const int blocks = (n_val + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemcpy(d_old_centroids, d_centroids, k * n_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    int* h_labels = new int[n_val];
    int iter = 0;

    auto end1 = std::chrono::high_resolution_clock::now();
    double avgTime1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
    printf("%lf\n", avgTime1);
    while (iter < max_iters) {

        
        findClosestCentroid(d_points, d_labels, n_val, n_dim, k, d_centroids);

        updateCentroids(d_points,d_labels, d_centroids, n_val, n_dim, k, d_centroids_sums, d_counts);
        iter++;
        cudaMemset(d_diffs, 0, k * n_dim * sizeof(float));

        if (hasConverged(d_centroids, d_old_centroids, d_diffs, k, n_dim, threshold)) {
            break;  // Exit loop if centroids have converged
        }

        // Swap the pointers
        cudaMemcpy(d_old_centroids, d_centroids, k * n_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(h_centroids, d_centroids, k * n_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Compute the elapsed time between start and stop
    cudaEventElapsedTime(&elapsedTime, start, stop);
    double avgTime = elapsedTime / iter;
    printf("%d,%lf\n", iter, avgTime);
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Cleanup
    cudaFree(d_centroids);
    cudaFree(d_old_centroids);
    cudaFree(data.label);
    cudaFree(d_diffs);


    return h_centroids;
}
