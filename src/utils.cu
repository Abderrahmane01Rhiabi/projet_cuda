#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// stb image bibliotheque image en un header
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// separateur
// charge une image depuis le disque avec stb image
// separateur
Image loadImage(const char* path) {
    Image img;
    img.data = stbi_load(path, &img.width, &img.height, &img.channels, 0);
    if (!img.data) {
        fprintf(stderr, "failed to load image : %s\n", path);
        exit(EXIT_FAILURE);
    }
    printf("image loaded : %s (%dx%d, %d channels)\n",
           path, img.width, img.height, img.channels);
    return img;
}

// separateur
// sauvegarde une image en png
// separateur
void saveImage(const char* path, Image img) {
    int ok = stbi_write_png(path, img.width, img.height,
                            img.channels, img.data,
                            img.width * img.channels);
    if (!ok) {
        fprintf(stderr, "failed to save image : %s\n", path);
        exit(EXIT_FAILURE);
    }
    printf("image saved : %s\n", path);
}

// separateur
// libere la memoire image
// separateur
void freeImage(Image* img) {
    if (img->data) {
        stbi_image_free(img->data);
        img->data = NULL;
    }
}

// separateur
// convertit rgb vers niveaux de gris methode luminosite
// la sortie est une image a 1 canal
// separateur
Image toGrayscale(Image img) {
    Image gray;
    gray.width    = img.width;
    gray.height   = img.height;
    gray.channels = 1;
    gray.data     = (unsigned char*)malloc(img.width * img.height);

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = (y * img.width + x) * img.channels;
            unsigned char r = img.data[idx + 0];
            unsigned char g = img.data[idx + 1];
            unsigned char b = img.data[idx + 2];
            // poids standard de luminosite
            gray.data[y * img.width + x] =
                (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    return gray;
}
