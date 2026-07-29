// The null request-model provider: the seam's twelve entries answered by
// a model with no speculator. Families without a drafter link this
// object instead of a model module, and the request tier runs plain
// decode with every speculative path fail-closed. The symbol-parity
// gate holds this file and the glm provider to the same extern set, so
// the seam cannot grow on one side without the other noticing.
#include <stdint.h>
#include "sparkpipe/spark_request_model.h"

uint32_t SparkRequestModelSpeculationIsEnabled(const SparkRequestApi *api)
{
	(void)api;
	return 0u;
}

uint32_t SparkRequestModelSpeculatorIsValid(const struct SparkRequestModelSpeculator *speculator)
{
	return speculator == 0;
}

uint32_t SparkRequestModelSlotCanSpeculate(const SparkRequestApi *api, const SparkRequestApiSlot *slot)
{
	(void)api;
	(void)slot;
	return 0u;
}

SparkStatus SparkRequestModelPrepareDraftForSlot(SparkRequestApi *api, SparkRequestApiSlot *slot)
{
	(void)api;
	(void)slot;
	return SPARK_STATUS_NOT_FOUND;
}

SparkStatus SparkRequestModelGetDraft(SparkRequestApi *api, uint64_t sequence_id, SparkRequestModelDraftResult *result)
{
	(void)api;
	(void)sequence_id;
	(void)result;
	return SPARK_STATUS_NOT_FOUND;
}

SparkStatus SparkRequestModelGetSlotSpeculativeDraft(SparkRequestApi *api, const SparkRequestApiSlot *slot, uint32_t preferred_source, SparkRequestModelDraftResult *draft_result, uint32_t *source_out)
{
	(void)api;
	(void)slot;
	(void)preferred_source;
	(void)draft_result;
	(void)source_out;
	return SPARK_STATUS_NOT_FOUND;
}

SparkStatus SparkRequestModelMarkVerifierTapsReady(SparkRequestApi *api, uint64_t request_id, uint64_t sequence_id, uint64_t sequence_position, uint64_t *tap_generation_out)
{
	(void)api;
	(void)request_id;
	(void)sequence_id;
	(void)sequence_position;
	(void)tap_generation_out;
	return SPARK_STATUS_NOT_FOUND;
}

uint32_t SparkRequestModelDefaultSpeculativeTokenCount(const SparkRequestApi *api)
{
	(void)api;
	return 0u;
}

SparkStatus SparkRequestModelCompleteVerify(SparkRequestApi *api, uint64_t sequence_id, const SparkRequestModelVerifyResult *verify_result)
{
	(void)api;
	(void)sequence_id;
	(void)verify_result;
	return SPARK_STATUS_NOT_FOUND;
}

SparkStatus SparkRequestModelResolveVerifierTokens(const uint32_t *draft_token_ids, uint32_t draft_token_count, const uint32_t *verifier_token_ids, uint32_t verifier_token_count, SparkRequestModelVerifyResult *verify_result)
{
	(void)draft_token_ids;
	(void)draft_token_count;
	(void)verifier_token_ids;
	(void)verifier_token_count;
	(void)verify_result;
	return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkRequestModelCancelSequence(SparkRequestApi *api, uint64_t sequence_id)
{
	(void)api;
	(void)sequence_id;
	return SPARK_STATUS_OK;
}

uint32_t SparkRequestModelMtpOutranksPlainDecode(const SparkRequestApi *api, uint32_t plain_request_count, uint32_t mtp_request_count)
{
	(void)api;
	(void)plain_request_count;
	(void)mtp_request_count;
	return 0u;
}
