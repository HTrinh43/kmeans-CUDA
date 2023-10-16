#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include "helpers.h"

struct squared_distance_functor {
    template <typename Tuple>
    __host__ __device__ float operator()(const Tuple& tuple) const {
        float diff = thrust::get<0>(tuple) - thrust::get<1>(tuple);
        return diff * diff;
    }
};

struct AddFunctor {
    const float* points;
    const int* labels;
    float* sum;
    int n_points, k, n_dim;

    AddFunctor(float* _points, int* _labels, float* _sum, int _n_points, int _k, int _n_dim)
        : points(_points), labels(_labels), sum(_sum), n_points(_n_points), k(_k), n_dim(_n_dim) {}

    __device__ void operator()(int idx) {
        int label = labels[idx];
        for (int d = 0; d < n_dim; ++d) {
            atomicAdd(&sum[label * n_dim + d], points[idx * n_dim + d]);
        }
    }
};

struct assign_labels_functor {
    const float* d_points;
    const float* d_centroids;
    int n_dim;
    int k;

    assign_labels_functor(const float* _points, const float* _centroids, int _n_dim, int _k) 
        : d_points(_points), d_centroids(_centroids), n_dim(_n_dim), k(_k) {}

    __device__ int operator()(int idx) const {
        float min_dist = FLT_MAX;
        int label = -1;

        for (int centroid_idx = 0; centroid_idx < k; centroid_idx++) {
            float dist = 0;
            for (int dim = 0; dim < n_dim; dim++) {
                float diff = d_points[idx * n_dim + dim] - d_centroids[centroid_idx * n_dim + dim];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                label = centroid_idx;
            }
        }
        
        return label;
    }
};


bool checkConvergence(const thrust::device_vector<float>& old_centroids, const thrust::device_vector<float>& new_centroids, float threshold) {
    thrust::device_vector<float> squared_differences(old_centroids.size());

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(old_centroids.begin(), new_centroids.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(old_centroids.end(), new_centroids.end())),
        squared_differences.begin(),
        squared_distance_functor()
    );

    float total_squared_distance = thrust::reduce(squared_differences.begin(), squared_differences.end());

    return sqrt(total_squared_distance) < threshold;
}

void updateCentroids(thrust::device_vector<float>& d_points, thrust::device_vector<int>& d_labels, 
                     thrust::device_vector<float>& d_centroids, int n_points, int n_dim, int k) {
    thrust::device_vector<float> d_centroid_sums(k * n_dim, 0.0f);
    thrust::device_vector<int> d_counts(k, 0);

    thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), 
                     thrust::counting_iterator<int>(n_points), 
                     AddFunctor(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_labels.data()), 
                                thrust::raw_pointer_cast(d_centroid_sums.data()), n_points, k, n_dim));

    thrust::sort(d_labels.begin(), d_labels.end());

    thrust::reduce_by_key(
        d_labels.begin(), d_labels.end(),
        thrust::constant_iterator<int>(1),
        thrust::make_discard_iterator(),
        d_counts.begin()
    );

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n_dim; ++j) {
            d_centroids[i * n_dim + j] = d_centroid_sums[i * n_dim + j] / d_counts[i];
        }
    }
}

void assignLabels(const thrust::device_vector<float>& d_points, 
                  thrust::device_vector<int>& d_labels, 
                  const thrust::device_vector<float>& d_centroids, int n_dim) {
    int k = d_centroids.size() / n_dim;
    int n_points = d_labels.size();

    thrust::counting_iterator<int> iter(0);
    assign_labels_functor functor(thrust::raw_pointer_cast(d_points.data()),
                                  thrust::raw_pointer_cast(d_centroids.data()),
                                  n_dim, k);
                                  
    thrust::transform(iter, iter + n_points, d_labels.begin(), functor);
}

float* kmeansThrust(Point_cu data, kmeans_args_t args) {
    int k = args.k;
    int n_points = data.size;
    int max_iter = args.m;
    float convergence_threshold = args.t;
    int n_dim = args.d;

    thrust::device_vector<float> d_points(data.features, data.features + n_points * n_dim);
    thrust::device_vector<int> d_labels(n_points, -1);
    thrust::device_vector<float> d_centroids(k * n_dim);
    thrust::device_vector<float> d_old_centroids(k * n_dim);

    float* initial_centroids_ptr = initializeCentroids(data, n_points, n_dim, k, args.s);
    thrust::copy(initial_centroids_ptr, initial_centroids_ptr + k * n_dim, d_centroids.begin());
    delete[] initial_centroids_ptr;

    int iter = 0;
    bool converged = false;

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    while (!converged && iter < max_iter) {
        assignLabels(d_points, d_labels, d_centroids, n_dim);
        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());
        updateCentroids(d_points, d_labels, d_centroids, n_points, n_dim, k);
        converged = checkConvergence(d_old_centroids, d_centroids, convergence_threshold);
        iter++;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avgTime = elapsedTime / iter;
    printf("%d,%lf\n", iter, avgTime);

    float* final_centroids = new float[k * n_dim];
    thrust::copy(d_centroids.begin(), d_centroids.end(), final_centroids);
    return final_centroids;
}
