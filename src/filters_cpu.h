#pragma once
#include "utils.h"

// fonctions de filtre cpu version reference
// entree en gris et sortie en gris

void cpuBoxBlur      (Image input, Image output);
void cpuSobel        (Image input, Image output);
void cpuGaussianBlur (Image input, Image output);
void cpuLoG          (Image input, Image output);
