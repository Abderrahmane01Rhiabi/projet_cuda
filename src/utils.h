#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

// separateur
// structure image largeur hauteur canaux donnees pixel
// separateur
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

// separateur
// macro de verification erreur cuda
// usage check cuda cudamalloc
// separateur
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "cuda error at %s:%d -> %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// separateur
// verifie la derniere erreur kernel apres kernel et sync
// separateur
#define CHECK_KERNEL()                                                          \
    do {                                                                        \
        CHECK_CUDA(cudaDeviceSynchronize());                                    \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "kernel error at %s:%d -> %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// separateur
// timer cpu avec std chrono
// separateur
typedef std::chrono::high_resolution_clock::time_point CpuTimer;

// demarre le timer cpu
inline CpuTimer cpuTimerStart() {
    return std::chrono::high_resolution_clock::now();
}

// arrete le timer cpu et renvoie le temps en ms
inline float cpuTimerStop(CpuTimer start) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    return elapsed.count();
}

// separateur
// timer gpu avec evenements cuda
// separateur
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

// cree et demarre le timer gpu
inline GpuTimer gpuTimerStart() {
    GpuTimer t;
    CHECK_CUDA(cudaEventCreate(&t.start));
    CHECK_CUDA(cudaEventCreate(&t.stop));
    CHECK_CUDA(cudaEventRecord(t.start, 0));
    return t;
}

// arrete le timer gpu et renvoie le temps en ms
inline float gpuTimerStop(GpuTimer t) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(t.stop, 0));
    CHECK_CUDA(cudaEventSynchronize(t.stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, t.start, t.stop));
    CHECK_CUDA(cudaEventDestroy(t.start));
    CHECK_CUDA(cudaEventDestroy(t.stop));
    return ms;
}

// separateur
// fonctions image definies dans utils cu
// separateur
Image loadImage(const char* path);
void  saveImage(const char* path, Image img);
void  freeImage(Image* img);

// convertit une image rgb en niveaux de gris
Image toGrayscale(Image img);

// borne une valeur entre 0 et 255
inline unsigned char clampUC(int val) {
    if (val < 0)   return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}
