#include "filters_gpu.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// separateur
// taille du bloc pour tous les kernels
// 16x16 donne 256 threads par bloc bonne occupation
// separateur
#define BLOCK_SIZE  16
#define NUM_STREAMS 4

// separateur
// outils device
// separateur
__device__ inline unsigned char dClamp(int val) {
    if (val < 0)   return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}

__device__ inline unsigned char dGetPixel(const unsigned char* data,
                                           int x, int y,
                                           int width, int height)
{
    x = max(0, min(x, width  - 1));
    y = max(0, min(y, height - 1));
    return data[y * width + x];
}

// separateur
// kernels en memoire constante lecture rapide
// separateur
__constant__ float cSobelX[9]  = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__ float cSobelY[9]  = {  1, 2, 1,  0, 0, 0, -1,-2,-1 };

__constant__ float cGaussian[25] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};

__constant__ float cLoG[81] = {
     0,  1,  1,   2,   2,   2,  1,  1,  0,
     1,  2,  4,   5,   5,   5,  4,  2,  1,
     1,  4,  5,   3,   0,   3,  5,  4,  1,
     2,  5,  3, -12, -24, -12,  3,  5,  2,
     2,  5,  0, -24, -40, -24,  0,  5,  2,
     2,  5,  3, -12, -24, -12,  3,  5,  2,
     1,  4,  5,   3,   0,   3,  5,  4,  1,
     1,  2,  4,   5,   5,   5,  4,  2,  1,
     0,  1,  1,   2,   2,   2,  1,  1,  0
};

#define GAUSS_NORM (1.0f / 273.0f)


// separateur
// flou boite
// separateur

__global__ void kernelBoxBlurNaive(const unsigned char* in,
                                    unsigned char* out,
                                    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += (float)dGetPixel(in, x+dx, y+dy, width, height);
    out[y * width + x] = dClamp((int)(sum / 9.0f));
}

#define TILE_BOX (BLOCK_SIZE + 2)
__global__ void kernelBoxBlurShared(const unsigned char* in,
                                     unsigned char* out,
                                     int width, int height)
{
    __shared__ unsigned char tile[TILE_BOX][TILE_BOX];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * BLOCK_SIZE + tx;
    int y  = blockIdx.y * BLOCK_SIZE + ty;

    tile[ty][tx] = dGetPixel(in, x-1, y-1, width, height);
    if (tx == BLOCK_SIZE-1)
        tile[ty][tx+1] = dGetPixel(in, x, y-1, width, height);
    if (ty == BLOCK_SIZE-1)
        tile[ty+1][tx] = dGetPixel(in, x-1, y, width, height);
    if (tx == BLOCK_SIZE-1 && ty == BLOCK_SIZE-1)
        tile[ty+1][tx+1] = dGetPixel(in, x, y, width, height);
    __syncthreads();

    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int dy = 0; dy <= 2; dy++)
        for (int dx = 0; dx <= 2; dx++)
            sum += (float)tile[ty+dy][tx+dx];
    out[y * width + x] = dClamp((int)(sum / 9.0f));
}

// version stream traite une bande horizontale
__global__ void kernelBoxBlurStream(const unsigned char* in,
                                     unsigned char* out,
                                     int width, int height, int yOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + yOffset;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += (float)dGetPixel(in, x+dx, y+dy, width, height);
    out[y * width + x] = dClamp((int)(sum / 9.0f));
}


// separateur
// sobel
// separateur

__global__ void kernelSobelNaive(const unsigned char* in,
                                  unsigned char* out,
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float gx = 0.0f, gy = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            int idx = (dy+1)*3 + (dx+1);
            gx += p * cSobelX[idx];
            gy += p * cSobelY[idx];
        }
    out[y * width + x] = dClamp((int)sqrtf(gx*gx + gy*gy));
}

#define TILE_SOBEL (BLOCK_SIZE + 2)
__global__ void kernelSobelShared(const unsigned char* in,
                                   unsigned char* out,
                                   int width, int height)
{
    __shared__ unsigned char tile[TILE_SOBEL][TILE_SOBEL];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * BLOCK_SIZE + tx;
    int y  = blockIdx.y * BLOCK_SIZE + ty;

    tile[ty][tx] = dGetPixel(in, x-1, y-1, width, height);
    if (tx == BLOCK_SIZE-1)
        tile[ty][tx+1] = dGetPixel(in, x, y-1, width, height);
    if (ty == BLOCK_SIZE-1)
        tile[ty+1][tx] = dGetPixel(in, x-1, y, width, height);
    if (tx == BLOCK_SIZE-1 && ty == BLOCK_SIZE-1)
        tile[ty+1][tx+1] = dGetPixel(in, x, y, width, height);
    __syncthreads();

    if (x >= width || y >= height) return;
    float gx = 0.0f, gy = 0.0f;
    for (int dy = 0; dy <= 2; dy++)
        for (int dx = 0; dx <= 2; dx++) {
            float p = (float)tile[ty+dy][tx+dx];
            int idx = dy*3 + dx;
            gx += p * cSobelX[idx];
            gy += p * cSobelY[idx];
        }
    out[y * width + x] = dClamp((int)sqrtf(gx*gx + gy*gy));
}

__global__ void kernelSobelStream(const unsigned char* in,
                                   unsigned char* out,
                                   int width, int height, int yOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + yOffset;
    if (x >= width || y >= height) return;

    float gx = 0.0f, gy = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            int idx = (dy+1)*3 + (dx+1);
            gx += p * cSobelX[idx];
            gy += p * cSobelY[idx];
        }
    out[y * width + x] = dClamp((int)sqrtf(gx*gx + gy*gy));
}


// separateur
// flou gaussien 5x5
// separateur

__global__ void kernelGaussNaive(const unsigned char* in,
                                  unsigned char* out,
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            sum += p * cGaussian[(dy+2)*5 + (dx+2)] * GAUSS_NORM;
        }
    out[y * width + x] = dClamp((int)sum);
}

#define TILE_GAUSS (BLOCK_SIZE + 4)
__global__ void kernelGaussShared(const unsigned char* in,
                                   unsigned char* out,
                                   int width, int height)
{
    __shared__ unsigned char tile[TILE_GAUSS][TILE_GAUSS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = blockIdx.x * BLOCK_SIZE - 2;
    int y0 = blockIdx.y * BLOCK_SIZE - 2;

    tile[ty][tx] = dGetPixel(in, x0+tx, y0+ty, width, height);
    if (tx < 4)
        tile[ty][tx+BLOCK_SIZE] = dGetPixel(in, x0+tx+BLOCK_SIZE, y0+ty, width, height);
    if (ty < 4)
        tile[ty+BLOCK_SIZE][tx] = dGetPixel(in, x0+tx, y0+ty+BLOCK_SIZE, width, height);
    if (tx < 4 && ty < 4)
        tile[ty+BLOCK_SIZE][tx+BLOCK_SIZE] = dGetPixel(in, x0+tx+BLOCK_SIZE, y0+ty+BLOCK_SIZE, width, height);
    __syncthreads();

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = 0; dy < 5; dy++)
        for (int dx = 0; dx < 5; dx++) {
            float p = (float)tile[ty+dy][tx+dx];
            sum += p * cGaussian[dy*5+dx] * GAUSS_NORM;
        }
    out[y * width + x] = dClamp((int)sum);
}

__global__ void kernelGaussStream(const unsigned char* in,
                                   unsigned char* out,
                                   int width, int height, int yOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + yOffset;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            sum += p * cGaussian[(dy+2)*5 + (dx+2)] * GAUSS_NORM;
        }
    out[y * width + x] = dClamp((int)sum);
}


// separateur
// laplacien du gaussien 9x9
// separateur

__global__ void kernelLoGNaive(const unsigned char* in,
                                unsigned char* out,
                                int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -4; dy <= 4; dy++)
        for (int dx = -4; dx <= 4; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            sum += p * cLoG[(dy+4)*9 + (dx+4)];
        }
    out[y * width + x] = dClamp((int)(sum / 8.0f) + 128);
}

#define TILE_LOG (BLOCK_SIZE + 8)
__global__ void kernelLoGShared(const unsigned char* in,
                                 unsigned char* out,
                                 int width, int height)
{
    __shared__ unsigned char tile[TILE_LOG][TILE_LOG];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = blockIdx.x * BLOCK_SIZE - 4;
    int y0 = blockIdx.y * BLOCK_SIZE - 4;

    tile[ty][tx] = dGetPixel(in, x0+tx, y0+ty, width, height);
    if (tx < 8)
        tile[ty][tx+BLOCK_SIZE] = dGetPixel(in, x0+tx+BLOCK_SIZE, y0+ty, width, height);
    if (ty < 8)
        tile[ty+BLOCK_SIZE][tx] = dGetPixel(in, x0+tx, y0+ty+BLOCK_SIZE, width, height);
    if (tx < 8 && ty < 8)
        tile[ty+BLOCK_SIZE][tx+BLOCK_SIZE] = dGetPixel(in, x0+tx+BLOCK_SIZE, y0+ty+BLOCK_SIZE, width, height);
    __syncthreads();

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = 0; dy < 9; dy++)
        for (int dx = 0; dx < 9; dx++) {
            float p = (float)tile[ty+dy][tx+dx];
            sum += p * cLoG[dy*9+dx];
        }
    out[y * width + x] = dClamp((int)(sum / 8.0f) + 128);
}

__global__ void kernelLoGStream(const unsigned char* in,
                                 unsigned char* out,
                                 int width, int height, int yOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + yOffset;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int dy = -4; dy <= 4; dy++)
        for (int dx = -4; dx <= 4; dx++) {
            float p = (float)dGetPixel(in, x+dx, y+dy, width, height);
            sum += p * cLoG[(dy+4)*9 + (dx+4)];
        }
    out[y * width + x] = dClamp((int)(sum / 8.0f) + 128);
}


// separateur
// wrappers host allocation copie liberation simple
// separateur

static void allocDevice(Image img, unsigned char** d_in, unsigned char** d_out) {
    int size = img.width * img.height;
    CHECK_CUDA(cudaMalloc(d_in,  size));
    CHECK_CUDA(cudaMalloc(d_out, size));
    CHECK_CUDA(cudaMemcpy(*d_in, img.data, size, cudaMemcpyHostToDevice));
}

static void freeDevice(Image output, unsigned char* d_in, unsigned char* d_out) {
    int size = output.width * output.height;
    CHECK_CUDA(cudaMemcpy(output.data, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

// separateur
// wrappers host avec streams
// coupe l image en bandes horizontales num streams
// chaque bande fait copie h2d kernel copie d2h en async
// toutes les bandes tournent en meme temps sur des streams separes
// separateur

typedef void (*StreamKernel)(const unsigned char*, unsigned char*,
                              int, int, int);

static void runWithStreams(Image input, Image output, StreamKernel kernel) {
    int size = input.width * input.height;

    // memoire pinned utile pour les transferts async
    unsigned char *h_in, *h_out;
    CHECK_CUDA(cudaMallocHost(&h_in,  size));
    CHECK_CUDA(cudaMallocHost(&h_out, size));
    memcpy(h_in, input.data, size);

    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

    int bandHeight = (input.height + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int yOffset    = i * bandHeight;
        int actualRows = min(bandHeight, input.height - yOffset);
        if (actualRows <= 0) break;

        int bandSize   = actualRows * input.width;
        int bandOffset = yOffset * input.width;

        // copie async de la bande host vers device
        CHECK_CUDA(cudaMemcpyAsync(d_in + bandOffset, h_in + bandOffset,
                                   bandSize, cudaMemcpyHostToDevice, streams[i]));

        // lance le kernel sur cette bande
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((input.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (actualRows  + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernel<<<grid, block, 0, streams[i]>>>(
            d_in, d_out, input.width, input.height, yOffset);

        // copie async du resultat device vers host
        CHECK_CUDA(cudaMemcpyAsync(h_out + bandOffset, d_out + bandOffset,
                                   bandSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    // attend tous les streams
    CHECK_CUDA(cudaDeviceSynchronize());
    memcpy(output.data, h_out, size);

    for (int i = 0; i < NUM_STREAMS; i++)
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}


// separateur
// fonctions wrapper publiques
// separateur

#define MAKE_WRAPPERS(name, naiveK, sharedK, streamK)                    \
void gpu##name(Image input, Image output) {                              \
    unsigned char *d_in, *d_out;                                         \
    allocDevice(input, &d_in, &d_out);                                   \
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);                                  \
    dim3 grid((input.width  + BLOCK_SIZE-1) / BLOCK_SIZE,               \
              (input.height + BLOCK_SIZE-1) / BLOCK_SIZE);              \
    naiveK<<<grid, block>>>(d_in, d_out, input.width, input.height);    \
    CHECK_CUDA(cudaDeviceSynchronize());                                  \
    freeDevice(output, d_in, d_out);                                     \
}                                                                        \
void gpu##name##Shared(Image input, Image output) {                      \
    unsigned char *d_in, *d_out;                                         \
    allocDevice(input, &d_in, &d_out);                                   \
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);                                  \
    dim3 grid((input.width  + BLOCK_SIZE-1) / BLOCK_SIZE,               \
              (input.height + BLOCK_SIZE-1) / BLOCK_SIZE);              \
    sharedK<<<grid, block>>>(d_in, d_out, input.width, input.height);   \
    CHECK_CUDA(cudaDeviceSynchronize());                                  \
    freeDevice(output, d_in, d_out);                                     \
}                                                                        \
void gpu##name##Stream(Image input, Image output) {                      \
    runWithStreams(input, output, (StreamKernel)streamK);                 \
}

MAKE_WRAPPERS(BoxBlur,      kernelBoxBlurNaive, kernelBoxBlurShared, kernelBoxBlurStream)
MAKE_WRAPPERS(Sobel,        kernelSobelNaive,   kernelSobelShared,   kernelSobelStream)
MAKE_WRAPPERS(GaussianBlur, kernelGaussNaive,   kernelGaussShared,   kernelGaussStream)
MAKE_WRAPPERS(LoG,          kernelLoGNaive,     kernelLoGShared,     kernelLoGStream)
