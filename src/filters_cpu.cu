#include "filters_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// separateur
// outil recupere pixel avec bord borne
// separateur
static inline unsigned char getPixel(Image img, int x, int y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.width)  x = img.width  - 1;
    if (y >= img.height) y = img.height - 1;
    return img.data[y * img.width + x];
}

// separateur
// outil applique un kernel de taille ksize x ksize
// ksize doit etre impair
// separateur
static void applyKernel(Image input, Image output,
                        const float* kernel, int ksize)
{
    int half = ksize / 2;
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    float pixel = (float)getPixel(input, x + kx, y + ky);
                    float coeff = kernel[(ky + half) * ksize + (kx + half)];
                    sum += pixel * coeff;
                }
            }
            output.data[y * input.width + x] = clampUC((int)sum);
        }
    }
}

// separateur
// flou boite avec kernel 3x3 uniforme
// separateur
void cpuBoxBlur(Image input, Image output) {
    // les 9 coefficients valent 1 sur 9
    const float k[9] = {
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f
    };
    applyKernel(input, output, k, 3);
}

// separateur
// sobel calcule la norme du gradient avec gx et gy
// separateur
void cpuSobel(Image input, Image output) {
    // gradient horizontal
    const float kx[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    // gradient vertical
    const float ky[9] = {
         1,  2,  1,
         0,  0,  0,
        -1, -2, -1
    };

    int half = 1;
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float gx = 0.0f;
            float gy = 0.0f;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    float p = (float)getPixel(input, x + dx, y + dy);
                    int idx = (dy + half) * 3 + (dx + half);
                    gx += p * kx[idx];
                    gy += p * ky[idx];
                }
            }
            // norme du gradient
            int mag = (int)sqrtf(gx * gx + gy * gy);
            output.data[y * input.width + x] = clampUC(mag);
        }
    }
}

// separateur
// flou gaussien avec kernel 5x5
// separateur
void cpuGaussianBlur(Image input, Image output) {
    // kernel gaussien 5x5 deja normalise
    const float k[25] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    // somme des valeurs egale a 273
    const float norm = 1.0f / 273.0f;

    int half = 2;
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    float p = (float)getPixel(input, x + dx, y + dy);
                    int idx = (dy + half) * 5 + (dx + half);
                    sum += p * k[idx] * norm;
                }
            }
            output.data[y * input.width + x] = clampUC((int)sum);
        }
    }
}

// separateur
// laplacien du gaussien avec kernel 9x9
// combine lissage gaussien et detection de bord
// separateur
void cpuLoG(Image input, Image output) {
    // kernel log 9x9 avec sigma 1 4 approximation
    const float k[81] = {
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

    int half = 4;
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sum = 0.0f;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    float p = (float)getPixel(input, x + dx, y + dy);
                    int idx = (dy + half) * 9 + (dx + half);
                    sum += p * k[idx];
                }
            }
            // decale pour voir le resultat log peut etre negatif
            int val = (int)(sum / 8.0f) + 128;
            output.data[y * input.width + x] = clampUC(val);
        }
    }
}
