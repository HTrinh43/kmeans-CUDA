
#include <iostream>
#include <io.h>
#include <kmeans.h>
#include <string.h>

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Setup pointer variable for input/output values
    int n_vals;
    std::vector<Point> dataSet;
    Point_cu dataSet2;
    // Pass empty pointers to file reader for filling
    read_file(&opts, &n_vals, dataSet, dataSet2);
    
    kmeans_args_t args = {opts.k, opts.d, opts.m, opts.t, opts.c, opts.s, opts.r};


    if (opts.r == 0) {
        // sequential
        std::vector<Point> centroids;
        centroids = kmeans(args, dataSet);
        if (opts.c){
            printPoints2(centroids);
        }
    }
    else{
        float* centroids;
        if (opts.r == 1){
            // basic CUDA
            centroids  = kmeansCUDA(dataSet2,args);
        }
        else if (opts.r == 2){
            // shmem
            centroids  = kmeansCUDAShare(dataSet2,args);
        }
        else{
            //thrust
            centroids = kmeansThrust(dataSet2, args);
        }
        if (opts.c){
            printPoints(args.k, args.d, centroids);
        }       
    }
    return 0;
}