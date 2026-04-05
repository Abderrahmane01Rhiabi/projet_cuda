#pragma once
#include "utils.h"

// fonctions de filtre gpu
// naive utilise seulement la memoire globale
// shared utilise la memoire partagee pour mieux aller
// stream utilise des streams cuda pour chevaucher transfert et calcul

void gpuBoxBlur            (Image input, Image output);
void gpuBoxBlurShared      (Image input, Image output);
void gpuBoxBlurStream      (Image input, Image output);

void gpuSobel              (Image input, Image output);
void gpuSobelShared        (Image input, Image output);
void gpuSobelStream        (Image input, Image output);

void gpuGaussianBlur       (Image input, Image output);
void gpuGaussianBlurShared (Image input, Image output);
void gpuGaussianBlurStream (Image input, Image output);

void gpuLoG                (Image input, Image output);
void gpuLoGShared          (Image input, Image output);
void gpuLoGStream          (Image input, Image output);
