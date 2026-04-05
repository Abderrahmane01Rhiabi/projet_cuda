# cuda image filtering project

## dependances

- nvidia cuda toolkit (nvcc)
- stb_image (single header, inclus dans src/)

## compilation

    make

## execution

    make run

ou

    ./image_filter images/lena.jpg

## structure du projet

    src/
      main.cu               -> point d'entree, lance tous les filtres et affiche les timings
      utils.h / utils.cu    -> chargement/sauvegarde images, macros erreurs cuda, timers
      filters_cpu.h / .cu   -> implementations de reference sur cpu
      filters_gpu.h / .cu   -> kernels gpu (naive + memoire partagee + streams)
      stb_image.h           -> bibliotheque de chargement d'images (single header)
      stb_image_write.h     -> bibliotheque de sauvegarde d'images (single header)
    images/                 -> images d'entree
    output/                 -> images de sortie generees
    Makefile

## filtres implementes

- box blur              (noyau 3x3)
- sobel                 (noyau 3x3, magnitude du gradient)
- gaussian blur         (noyau 5x5)
- laplacian of gaussian (noyau 9x9)

## optimisations gpu

- version naive         : memoire globale uniquement
- version shared        : memoire partagee + memoire constante
- version streams       : cuda streams avec copies asynchrones (memoire pinned)

## gpu teste

- nvidia geforce rtx 3060 laptop, cuda 12.8, wsl2