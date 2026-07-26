// Compile the whole of a model's unity build with no CUDA toolchain.
//
// Defines the CUDA keywords away and includes the model's unity.cu. Inline PTX
// still parses, its constraints do not resolve, and that noise is filtered by
// tools/gates.sh; what this catches is everything else - template errors, wrong
// argument counts, undeclared symbols, and every static_assert in the geometry.
//
// Include order matters and cost an hour once: a stale copy of kernels/ inside
// the shim directory shadowed the real headers because -I<shim> preceded -I<root>.
// The tree comes first on the command line for that reason.
#include "shim.h"
static inline float __half2float(unsigned short) { return 0.0f; }
static inline unsigned short __ushort_as_half(unsigned short v) { return v; }
struct float2_ { float x, y; };
#define float2 float2_
#include LM_UNITY
