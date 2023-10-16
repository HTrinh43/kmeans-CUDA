#include "helpers.h"
#include <iostream>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}


// Helper function to mark a number as used
void mark_as_used(bool used[], int num) {
    used[num] = true;
}

// Check if a number is already used
bool is_used(bool used[], int num) {
    return used[num];
}

// Generate a list of `n` unique random numbers
int* generate_unique_numbers(int n, int list_size, int seed) {
    kmeans_srand(seed);
    int *arr = (int *)malloc(n * sizeof(int));
    bool *used = (bool *)calloc((list_size + 1), sizeof(bool));
    if (arr == NULL || used == NULL) {
        // Memory allocation failed
        free(arr); // Safe to free a NULL pointer
        free(used); // Safe to free a NULL pointer
        return NULL;
    }
    int count = 0;
    while (count < n) {
        int randValue = kmeans_rand()%(list_size);
        if (!is_used(used, randValue)) {
            arr[count] = randValue;
            mark_as_used(used, randValue);
            count++;
        }
    }

    free(used);
    return arr;
}


void printPoints(int n, int d, float* centroids){
    for (int i = 0; i < n; i++) {
        printf("%d ", i);
        for (int j = 0; j < d; j++) {
            // Print each dimension separated by a space
            printf("%lf ", centroids[i * d + j]);
        }
        // Move to the next line for the next centroid
        printf("\n");
    }
}

void printPoints2(std::vector<Point> centroids){
    size_t k = centroids.size();
    size_t d = centroids[0].features.size();
    for (size_t i = 0; i < k; i++) {
        printf("%d ", (int)i);
        for (size_t j = 0; j < d; j++) {
            // Print each dimension separated by a space
            printf("%lf ", centroids[i].features[j]);
        }
        // Move to the next line for the next centroid
        printf("\n");
    }
}



float* initializeCentroids(Point_cu data, int n_val, int n_dim, int k, int seed) {
    int *uniqueNums = generate_unique_numbers(k, n_val,seed);
    // Allocate memory for the centroids
    float* centroids = (float*)malloc(k * n_dim * sizeof(float));
    if (centroids == NULL) {
        // Memory allocation failed
        free(uniqueNums);
        return NULL;
    }

    // Copy the features of the randomly chosen points to the centroids
    for (int i = 0; i < k; i++) {
        int dataIndex = uniqueNums[i];
        for (int j = 0; j < n_dim; j++) {
            centroids[i * n_dim + j] = data.features[dataIndex * n_dim + j];
        }
    }
    
    //printPoints(k, n_dim, centroids);
    free(uniqueNums);
    return centroids;
}