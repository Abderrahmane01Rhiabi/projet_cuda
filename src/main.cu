#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "filters_cpu.h"
#include "filters_gpu.h"

// separateur
// affiche le temps
// separateur
void printTiming(const char* label, float ms) {
    printf("  %-30s : %.3f ms\n", label, ms);
}

// separateur
// warmup gpu avec kernel vide pour initialiser le contexte cuda
// separateur
__global__ void kernelWarmup() {}

void gpuWarmup() {
    kernelWarmup<<<1, 1>>>();
    cudaDeviceSynchronize();
}

// separateur
// lance un filtre cpu gpu naive gpu shared gpu stream
// separateur
void runFilter(const char* name,
               Image input,
               void (*cpuFilter)   (Image, Image),
               void (*gpuFilter)   (Image, Image),
               void (*gpuShared)   (Image, Image),
               void (*gpuStream)   (Image, Image),
               const char* outPrefix)
{
    printf("\n=== filter : %s ===\n", name);

    int sz = input.width * input.height;
    Image outCpu    = { input.width, input.height, 1, (unsigned char*)malloc(sz) };
    Image outGpu    = { input.width, input.height, 1, (unsigned char*)malloc(sz) };
    Image outShared = { input.width, input.height, 1, (unsigned char*)malloc(sz) };
    Image outStream = { input.width, input.height, 1, (unsigned char*)malloc(sz) };

    // version cpu
    CpuTimer ct = cpuTimerStart();
    cpuFilter(input, outCpu);
    float cpuMs = cpuTimerStop(ct);
    printTiming("cpu", cpuMs);

    // version gpu naive
    GpuTimer gt = gpuTimerStart();
    gpuFilter(input, outGpu);
    float gpuMs = gpuTimerStop(gt);
    printTiming("gpu naive", gpuMs);

    // version gpu memoire partagee
    GpuTimer gs = gpuTimerStart();
    gpuShared(input, outShared);
    float sharedMs = gpuTimerStop(gs);
    printTiming("gpu shared memory", sharedMs);

    // version gpu avec streams
    GpuTimer gst = gpuTimerStart();
    gpuStream(input, outStream);
    float streamMs = gpuTimerStop(gst);
    printTiming("gpu streams", streamMs);

    // gains de vitesse
    printf("  speedup cpu/gpu naive    : %.1fx\n", cpuMs / gpuMs);
    printf("  speedup cpu/gpu shared   : %.1fx\n", cpuMs / sharedMs);
    printf("  speedup cpu/gpu streams  : %.1fx\n", cpuMs / streamMs);

    // sauvegarde les resultats
    char path[256];
    snprintf(path, sizeof(path), "%s_cpu.png",    outPrefix);
    saveImage(path, outCpu);
    snprintf(path, sizeof(path), "%s_gpu.png",    outPrefix);
    saveImage(path, outGpu);
    snprintf(path, sizeof(path), "%s_stream.png", outPrefix);
    saveImage(path, outStream);

    free(outCpu.data);
    free(outGpu.data);
    free(outShared.data);
    free(outStream.data);
}

// separateur
// main
// separateur
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage : %s <image_path>\n", argv[0]);
        return 1;
    }

    printf("warming up gpu...\n");
    gpuWarmup();
    printf("gpu ready.\n");

    Image rgb  = loadImage(argv[1]);
    Image gray = toGrayscale(rgb);
    freeImage(&rgb);

    printf("\nimage size : %dx%d\n", gray.width, gray.height);

    runFilter("box blur",
              gray, cpuBoxBlur, gpuBoxBlur, gpuBoxBlurShared, gpuBoxBlurStream,
              "output/out_boxblur");

    runFilter("sobel",
              gray, cpuSobel, gpuSobel, gpuSobelShared, gpuSobelStream,
              "output/out_sobel");

    runFilter("gaussian blur",
              gray, cpuGaussianBlur, gpuGaussianBlur, gpuGaussianBlurShared, gpuGaussianBlurStream,
              "output/out_gaussian");

    runFilter("laplacian of gaussian",
              gray, cpuLoG, gpuLoG, gpuLoGShared, gpuLoGStream,
              "output/out_log");

    freeImage(&gray);
    printf("\ndone.\n");
    return 0;
}
