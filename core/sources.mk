SPARKPIPE_CORE_SUPPORT_SOURCES := \
    src/spark_status.c \
    runtime/filesystem.c \
    runtime/json.c \
    src/spark_sha256.c

SPARKPIPE_CORE_COMPILER_SOURCES := \
    runtime/pack/model_description.c \
    runtime/pack/module_library.c \
    runtime/pack/driver_compiler.c

SPARKPIPE_CORE_RUNTIME_SOURCES := \
    src/spark_driver_loader.c \
    src/spark_orchestrator.c
