REPOSITORY_ROOT := $(abspath ../..)
NVCC ?= nvcc
AR ?= ar
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -std=c++17 -O3 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/glm52_dspark_draft_backend
BACKEND_OBJECT := $(BUILD_DIRECTORY)/spark_glm52_dspark_draft_backend.o
BACKEND_ARCHIVE := $(BUILD_DIRECTORY)/libglm52_dspark_draft_backend.a

.PHONY: archive clean

archive: $(BACKEND_ARCHIVE)

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

$(BACKEND_OBJECT): source/spark_glm52_dspark_draft_backend.cu include/sparkpipe/spark_glm52_dspark_draft_backend.h | $(BUILD_DIRECTORY)
	$(NVCC) -arch=$(CUDA_ARCH) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -c $< -o $@

$(BACKEND_ARCHIVE): $(BACKEND_OBJECT)
	rm -f $@
	$(AR) rcs $@ $^

clean:
	rm -rf $(BUILD_DIRECTORY)
