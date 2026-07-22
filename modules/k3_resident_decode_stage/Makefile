REPOSITORY_ROOT := $(abspath ../..)
CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_121a
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O3
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -lineinfo -arch=$(CUDA_ARCH)
MODULE_LIBRARY_ROOT ?= $(REPOSITORY_ROOT)/build/module_library
MAX_STAGE_MICROSECONDS ?= 0

# Runtime configuration. The module itself requires every one of these to be
# present and parsable and never substitutes a default, so the defaults live
# here where they are visible and overridable.
K3_STAGE_PACK ?= $(BUILD_DIRECTORY)/k3_synthetic_stage.k3sp
K3_MAX_LANES ?= 4
K3_MAX_CONTEXT_TOKENS ?= 4096
K3_PIPELINE_SLOTS ?= 2
K3_PACK_SEED ?= 20260727

BUILD_DIRECTORY := $(REPOSITORY_ROOT)/build/modules/k3_resident_decode_stage
HOST_OBJECT := $(BUILD_DIRECTORY)/spark_k3_resident_decode_stage_module.o
CUDA_OBJECT := $(BUILD_DIRECTORY)/spark_k3_resident_decode_stage_cuda.o
MODULE_ARCHIVE := $(BUILD_DIRECTORY)/libk3_resident_decode_stage.a
PACK_SYNTHESIZE_TOOL := $(BUILD_DIRECTORY)/k3_pack_synthesize
CPU_REFERENCE_TOOL := $(BUILD_DIRECTORY)/spark_k3_reference
PACKAGE_DIRECTORY ?= $(REPOSITORY_ROOT)/build/packages/k3_resident_decode_stage

MODULE_IDENTIFIER := spark.k3.resident_decode_stage.mxfp4.h7168.l72.e896.k16.kda128.mla512.rv256.v1
MODULE_TARGET := cuda.sm121.k3.resident_decode_stage.mxfp4
VALIDATION_RECIPE := k3.resident_decode_stage.$(CUDA_ARCH).cpu_reference.max_us_$(MAX_STAGE_MICROSECONDS).v1

K3_RUNTIME_ENV := \
	K3_STAGE_PACK=$(K3_STAGE_PACK) \
	K3_MAX_LANES=$(K3_MAX_LANES) \
	K3_MAX_CONTEXT_TOKENS=$(K3_MAX_CONTEXT_TOKENS) \
	K3_PIPELINE_SLOTS=$(K3_PIPELINE_SLOTS)

.PHONY: all archive tools pack reference validate publish package clean require_sm121a

all: archive tools

$(BUILD_DIRECTORY):
	mkdir -p $(BUILD_DIRECTORY)

# The KDA chunk kernel is built for the Blackwell consumer target the Spark
# nodes run; a different arch would silently change the shared memory budget
# the kernel's static plan is proven against, so it is an error, not a warning.
require_sm121a:
	@if [ "$(CUDA_ARCH)" != "sm_121a" ]; then \
		echo "k3_resident_decode_stage requires CUDA_ARCH=sm_121a (got $(CUDA_ARCH))" >&2; \
		exit 1; \
	fi

$(CUDA_OBJECT): source/spark_k3_resident_decode_stage_cuda.cu | $(BUILD_DIRECTORY)
	$(NVCC) $(NVCCFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-Xcompiler -Wall,-Wextra,-fPIC,-fvisibility=hidden -MMD -MP -c $< -o $@

$(HOST_OBJECT): source/spark_k3_resident_decode_stage_module.c | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) -fPIC -I$(REPOSITORY_ROOT)/include -Iinclude -Isource \
		-I$(CUDA_HOME)/include -MMD -MP -c $< -o $@

$(MODULE_ARCHIVE): $(HOST_OBJECT) $(CUDA_OBJECT)
	$(AR) rcs $@ $(HOST_OBJECT) $(CUDA_OBJECT)

archive: require_sm121a $(MODULE_ARCHIVE)

$(PACK_SYNTHESIZE_TOOL): tools/k3_pack_synthesize.c | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $< -o $@ -lm

$(CPU_REFERENCE_TOOL): validation/spark_k3_reference.c | $(BUILD_DIRECTORY)
	$(CC) $(CFLAGS) -I$(REPOSITORY_ROOT)/include -Iinclude -Isource $< -o $@ -lm

tools: $(PACK_SYNTHESIZE_TOOL) $(CPU_REFERENCE_TOOL)

# A synthetic pack carries the real geometry with reproducible pseudo-random
# weights. It exercises the loader, the shape table and the layer walk; it
# says nothing about output quality.
pack: $(PACK_SYNTHESIZE_TOOL)
	$(PACK_SYNTHESIZE_TOOL) --output $(K3_STAGE_PACK) --seed $(K3_PACK_SEED)

# The cpu reference proves the datapath contracts that do not need a device:
# the KDA chunk recurrence, the AttnRes mixture, the router and SiTU.
reference: $(CPU_REFERENCE_TOOL)
	$(CPU_REFERENCE_TOOL)

validate: archive reference
	@echo "k3_resident_decode_stage: host and cuda units built, cpu reference passed"
	@echo "device execution requires a GB10 node; see STATUS.md"

publish: require_sm121a archive
	$(MAKE) -C $(REPOSITORY_ROOT) build/sparkpipe_module_publish
	$(K3_RUNTIME_ENV) \
		$(REPOSITORY_ROOT)/build/sparkpipe_module_publish \
		--library $(MODULE_LIBRARY_ROOT) \
		--module $(MODULE_IDENTIFIER) \
		--target $(MODULE_TARGET) \
		--link-unit $(MODULE_ARCHIVE) \
		--recipe $(VALIDATION_RECIPE) \
		--initialize SparkK3ResidentDecodeStageInitialize \
		--execute SparkK3ResidentDecodeStageExecute \
		--admit SparkK3ResidentDecodeStageAdmit \
		--snapshot SparkK3ResidentDecodeStageSnapshot \
		--destroy SparkK3ResidentDecodeStageDestroy

package: publish
	mkdir -p $(PACKAGE_DIRECTORY)
	cp $(MODULE_ARCHIVE) $(PACKAGE_DIRECTORY)/

clean:
	rm -rf $(BUILD_DIRECTORY)

-include $(BUILD_DIRECTORY)/*.d
