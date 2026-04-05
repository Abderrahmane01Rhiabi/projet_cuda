# -------------------------------------------------------
# makefile pour projet de filtre image cuda
# usage make construit
# usage make clean supprime le binaire
# usage make run construit et lance
# -------------------------------------------------------

NVCC      = nvcc
NVCCFLAGS = -O2 -std=c++14 -Xcompiler -Wall -Isrc

SRCS = src/main.cu src/utils.cu src/filters_cpu.cu src/filters_gpu.cu

TARGET = image_filter

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET) images/lena.jpg

clean:
	rm -f $(TARGET)

.PHONY: all run clean
