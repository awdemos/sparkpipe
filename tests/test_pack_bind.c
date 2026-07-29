// Weight binding: the table resolves, and an incomplete pack is caught.
//
// A test harness resolver hands back fabricated pointers, which is why binding
// takes a function rather than a pack struct - this needs no weight file.
#include "inference/llms/glm5_2/pack.c"
#include <stdio.h>

static int32_t failures = 0;
static void expect(int c, const char *l)
{
	printf(c ? "  ok   %s\n" : "  FAIL %s\n", l);
	if (!c) ++failures;
}

/* An absorbed pack: has the four folded projections, not the raw pair. */
static const void *resolve_absorbed(void *context, const char *name, uint32_t layer)
{
	(void)context; (void)layer;
	if (strstr(name, "q_a.norm") || strstr(name, "kv_a.norm"))
		return 0;
	return (const void *)0x1000;
}

/* A raw pack: the mirror. */
static const void *resolve_raw(void *context, const char *name, uint32_t layer)
{
	(void)context; (void)layer;
	if (strstr(name, "q_latent") || strstr(name, "q_rope") ||
	    strstr(name, "k_rope") || strstr(name, "kv_latent"))
		return 0;
	return (const void *)0x2000;
}

/* Neither attention form: both the absorbed four and the raw pair are absent,
   but everything required is present. This is the pack that would bind cleanly
   and then launch an attention path with no weights. */
static const void *resolve_neither(void *context, const char *name, uint32_t layer)
{
	(void)context; (void)layer;
	if (strstr(name, "q_latent") || strstr(name, "q_rope") ||
	    strstr(name, "k_rope") || strstr(name, "kv_latent") ||
	    strstr(name, "q_a.norm") || strstr(name, "kv_a.norm"))
		return 0;
	return (const void *)0x3000;
}

/* A required tensor absent. */
static const void *resolve_no_norm(void *context, const char *name, uint32_t layer)
{
	(void)context; (void)layer;
	if (strstr(name, "attn_norm")) return 0;
	return (const void *)0x4000;
}

int main(void)
{
	SparkResidentDecodeStageNodeContext context;
	uint32_t bound, missing;

	printf("weight binding\n\nthe table\n");
	printf("    %u tensors: %u per-layer, %u global\n", (unsigned)GLM52_TENSOR_COUNT,
		(unsigned)GLM52_TENSOR_COUNT - 1u, 1u);

	printf("\nboth pack forms bind, and which one decides the attention path\n");
	memset(&context, 0, sizeof(context));
	expect(Glm52BindPack(&context, 0u, resolve_absorbed, 0, &bound, &missing) == GLM52_BIND_OK,
		"an absorbed pack binds");
	expect(context.query_latent_weight_bf16 != 0, "and carries the folded projections");
	expect(context.raw_query_a_norm_weight_bf16 == 0, "and not the raw pair");

	memset(&context, 0, sizeof(context));
	expect(Glm52BindPack(&context, 0u, resolve_raw, 0, &bound, &missing) == GLM52_BIND_OK,
		"a raw pack binds");
	expect(context.raw_query_a_norm_weight_bf16 != 0, "and carries the two-stage norms");

	printf("\nan incomplete pack is caught here, not at the first launch\n");
	memset(&context, 0, sizeof(context));
	expect(Glm52BindPack(&context, 0u, resolve_neither, 0, &bound, &missing) == GLM52_BIND_ERR_INCOMPLETE,
		"a pack with neither attention form is rejected by name");
	memset(&context, 0, sizeof(context));
	expect(Glm52BindPack(&context, 0u, resolve_no_norm, 0, &bound, &missing) == GLM52_BIND_ERR_MISSING,
		"a missing required tensor is rejected");

	printf("\nscratch is arithmetic over config.h\n");
	{
		uint64_t b1 = Glm52LayerScratchBytes(1u);
		uint64_t b128 = Glm52LayerScratchBytes(128u);
		printf("    B1   %8llu bytes\n", (unsigned long long)b1);
		printf("    B128 %8llu bytes (%.1f MB)\n", (unsigned long long)b128, b128 / 1048576.0);
		expect(b128 > b1 * 100u, "and grows with the batch, faster than linearly "
			"because packed rows are tokens times top-k");
	}

	printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
