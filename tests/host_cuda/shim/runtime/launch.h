#pragma once
// launch.h carries the plan machinery the real GEMM needs; the recorder does
// not, and the error codes it defines live in the gemm shim so a caller sees
// exactly one definition.
#include <stdint.h>

// THE REAL CODES, COPIED, NOT CHOSEN. runtime/launch.h assigns these and a
// harness that made up its own would let a kernel return -1 where the tree
// means -41 and call it the same failure.
#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)
