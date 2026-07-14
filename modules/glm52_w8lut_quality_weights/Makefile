CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
NVFLAGS ?= -O3 -arch=$(CUDA_ARCH)
PYTHON ?= python3
BUILD_DIR ?= build
SOURCE_DIR := source
REFERENCE_TEST := $(BUILD_DIR)/test_w8lut_ref
REFERENCE_DUMP := $(BUILD_DIR)/w8lut_dump
GPU_TEST := $(BUILD_DIR)/test_w8lut_gpu

.PHONY: clean crosscheck test_gpu test_ref

$(BUILD_DIR):
	mkdir -p $@

$(REFERENCE_TEST): $(SOURCE_DIR)/w8lut_ref.c $(SOURCE_DIR)/test_w8lut_ref.c $(SOURCE_DIR)/w8lut.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SOURCE_DIR) -o $@ $(SOURCE_DIR)/w8lut_ref.c $(SOURCE_DIR)/test_w8lut_ref.c -lm

$(REFERENCE_DUMP): $(SOURCE_DIR)/w8lut_dump.c $(SOURCE_DIR)/w8lut_ref.c $(SOURCE_DIR)/w8lut.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SOURCE_DIR) -o $@ $(SOURCE_DIR)/w8lut_dump.c $(SOURCE_DIR)/w8lut_ref.c

$(GPU_TEST): $(SOURCE_DIR)/test_w8lut_gpu.cu $(SOURCE_DIR)/w8lut_kernels.cu $(SOURCE_DIR)/w8lut_ref.c $(SOURCE_DIR)/w8lut.h | $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) -I$(SOURCE_DIR) -o $@ $(SOURCE_DIR)/test_w8lut_gpu.cu $(SOURCE_DIR)/w8lut_kernels.cu $(SOURCE_DIR)/w8lut_ref.c

test_ref: $(REFERENCE_TEST)
	$(REFERENCE_TEST)

crosscheck: $(REFERENCE_DUMP)
	W8LUT_DUMP=$(abspath $(REFERENCE_DUMP)) $(PYTHON) $(SOURCE_DIR)/test_crosscheck.py

test_gpu: $(GPU_TEST)
	$(GPU_TEST)

clean:
	rm -rf $(BUILD_DIR)
