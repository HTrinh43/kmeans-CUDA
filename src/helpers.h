#ifndef HELPERS_H
#define HELPERS_H

// function declarations


#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct Point_cu {
    float* features;
    int* label;
    int size;
};

struct Point {
    std::vector<float> features;
    int label;
};

struct kmeans_args_t {
    int k;
    int d;
    int m;
    double t;
    bool c;
    int s;
    int r;
};
int* generate_unique_numbers(int n, int list_size, int seed);
float* initializeCentroids(Point_cu data, int n_val, int n_dim, int k, int seed);
void printPoints(int n, int d, float* centroids);
void printPoints2(std::vector<Point> centroids);
#endif // HELPERS_H