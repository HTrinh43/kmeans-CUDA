#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <helpers.h>
#include <cmath>


std::vector<Point> kmeans(kmeans_args_t& args,std::vector<Point>& data);
float* kmeansCUDA(Point_cu data, kmeans_args_t args);
float* kmeansCUDAShare(Point_cu data, kmeans_args_t args);
float* kmeansThrust(Point_cu data, kmeans_args_t args);
// float* kmeansThrust(Point_cu data, kmeans_args_t args);
#endif