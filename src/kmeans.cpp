#include <kmeans.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <limits>


double euclideanDistance(const Point& a, const Point& b) {
    double sum = 0;
    for (size_t i = 0; i < a.features.size(); ++i) {
        double diff = a.features[i] - b.features[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

bool converged(const std::vector<Point>& newCentroids, const std::vector<Point>& oldCentroids, double threshold) {
    for (size_t i = 0; i < newCentroids.size(); ++i) {
        if (euclideanDistance(newCentroids[i], oldCentroids[i]) > threshold) {
            return false;
        }
    }
    return true;
}

std::vector<Point> initializeCentroids(const std::vector<Point>& data, int k, int seed) {
    std::vector<Point> centroids;
    int* uniqueNums = generate_unique_numbers(k, data.size(), seed);
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[uniqueNums[i]]);
    }
    free(uniqueNums);
    return centroids;
}


void assignLabels(std::vector<Point>& data, const std::vector<Point>& centroids) {
    for (Point& point : data) {
        double minDist = std::numeric_limits<double>::max();
        int bestLabel = -1;

        for (size_t i = 0; i < centroids.size(); ++i) {
            double dist = euclideanDistance(point, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                bestLabel = i;
            }
        }
        point.label = bestLabel;
    }
}

std::vector<Point> computeCentroids(const std::vector<Point>& data, int k, int numFeatures) {
    std::vector<Point> centroids(k, {std::vector<float>(numFeatures, 0.0), -1});
    std::vector<int> counts(k, 0);

    for (const Point& point : data) {
        for (int feature = 0; feature < numFeatures; ++feature) {
            centroids[point.label].features[feature] += point.features[feature];
        }
        counts[point.label]++;
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            centroids[i].features[j] /= counts[i];
        }
    }

    return centroids;
}

std::vector<Point> kmeans(kmeans_args_t& args, std::vector<Point>& data) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Point> centroids = initializeCentroids(data, args.k, args.s);
    std::vector<Point> oldCentroids;

    int iter = 0;
    while (iter < args.m) {
        assignLabels(data, centroids);
        oldCentroids = centroids;
        centroids = computeCentroids(data, args.k, data[0].features.size());
        iter++;
        if (converged(oldCentroids, centroids, args.t)) {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double avgTime = std::chrono::duration<double, std::milli>(end - start).count() / iter;
    printf("%d,%lf\n", iter, avgTime);

    return centroids;
}
