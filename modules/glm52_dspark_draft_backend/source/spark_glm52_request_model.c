// The request model seam: glm's draft/verify mechanics, moved behind the
// linker exactly like the stage validation tier. request.c declares these
// seven entries with neutral names and no model types; this module defines
// them against the real speculator. The opaque configuration pointer is
// cast back to its concrete type here and nowhere else.
#include <string.h>
#include "sparkpipe/spark_request_model.h"
#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_mtp_tree.h"

uint64_t SparkRequestApiMtpResolvedRequestCount(const SparkRequestApi *api, uint64_t *committed_token_count_out);
uint32_t SparkRequestApiSlotHasRealtimePriority(const SparkRequestApiSlot *slot);
uint32_t SparkRequestApiSlotRemainingDecodeBudget(const SparkRequestApiSlot *slot);

_Static_assert(SPARK_GLM52_MODEL_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ==
	SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS,
	"the neutral speculative capacity mirrors the model value");

static inline SparkGlm52DsparkSpeculator *SparkRequestModelSpec(const SparkRequestApi *api)
{
	return (SparkGlm52DsparkSpeculator *)api->model_speculator;
}

uint32_t SparkRequestModelSpeculationIsEnabled(
    const SparkRequestApi *api)
{
    return (api->configuration_flags &
        SPARK_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE) != 0u &&
        SparkRequestModelSpec(api) != 0;
}


static SparkStatus SparkRequestModelGetSlotDsparkDraft(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    SparkGlm52DsparkDraftResult *draft_result)
{
    if (!SparkRequestModelSpeculationIsEnabled(api) ||
        slot == 0 || draft_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52DsparkGetDraft(
        SparkRequestModelSpec(api),
        slot->sequence_id,
        draft_result);
}

static SparkStatus SparkRequestModelGetSlotMtpDraft(
    const SparkRequestApiSlot *slot,
    SparkGlm52DsparkDraftResult *draft_result)
{
    uint32_t token_index;

    if (slot == 0 || draft_result == 0 ||
        slot->mtp_draft_token_count == 0u ||
        slot->mtp_draft_token_count >
            SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(draft_result, 0, sizeof(*draft_result));
    draft_result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    draft_result->descriptor_bytes =
        SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    draft_result->token_count = slot->mtp_draft_token_count;
    for (token_index = 0u;
         token_index < slot->mtp_draft_token_count;
         ++token_index)
    {
        draft_result->token_ids[token_index] =
            slot->mtp_draft_token_ids[token_index];
        draft_result->confidence_milli[token_index] =
            SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRequestModelGetSlotSpeculativeDraft(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t preferred_source,
    SparkGlm52DsparkDraftResult *draft_result,
    uint32_t *source_out)
{
    SparkStatus status;

    if (draft_result == 0 || source_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *source_out = 0u;

    if (preferred_source != SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_DRAFTER)
    {
        status = SparkRequestModelGetSlotMtpDraft(slot, draft_result);
        if (status == SPARK_STATUS_OK)
        {
            *source_out = SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP;
            return SPARK_STATUS_OK;
        }
        if (preferred_source == SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP)
        {
            return status;
        }
        status = SparkRequestModelGetSlotDsparkDraft(api, slot, draft_result);
        if (status == SPARK_STATUS_OK)
        {
            *source_out = SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_DRAFTER;
            return SPARK_STATUS_OK;
        }
        return status;
    }

    status = SparkRequestModelGetSlotDsparkDraft(api, slot, draft_result);
    if (status == SPARK_STATUS_OK)
    {
        *source_out = SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_DRAFTER;
        return SPARK_STATUS_OK;
    }
    status = SparkRequestModelGetSlotMtpDraft(slot, draft_result);
    if (status == SPARK_STATUS_OK)
    {
        *source_out = SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP;
        return SPARK_STATUS_OK;
    }
    return status;
}

static uint64_t SparkRequestModelMtpExpectedCommittedTokensScaled(
    const SparkRequestApi *api)
{
    uint64_t committed_token_count;
    uint64_t resolved_request_count;
    uint64_t weighted_committed_token_count;
    uint64_t weighted_request_count;

    resolved_request_count = SparkRequestApiMtpResolvedRequestCount(
        api,
        &committed_token_count);
    weighted_committed_token_count =
        committed_token_count +
        ((uint64_t)SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT *
         SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT);
    weighted_request_count =
        resolved_request_count +
        SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT;
    return weighted_committed_token_count *
        SPARK_GLM52_REQUEST_API_MTP_UTILITY_SCALE /
        weighted_request_count;
}

uint32_t SparkRequestModelMtpOutranksPlainDecode(
    const SparkRequestApi *api,
    uint32_t plain_request_count,
    uint32_t mtp_request_count)
{
    uint64_t mtp_expected_tokens_scaled;
    uint64_t mtp_work_ns;
    uint64_t plain_work_ns;
    SparkStatus status;

    if (api == 0 || api->scheduler == 0 ||
        plain_request_count == 0u || mtp_request_count == 0u)
    {
        return 0u;
    }
    if ((api->configuration_flags &
            SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE) != 0u)
    {
        return 1u;
    }
    status = SparkSchedulerEstimateDecodeWorkNs(
        api->scheduler,
        plain_request_count,
        1u,
        api->decode_execution_row_capacity,
        &plain_work_ns);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    status = SparkSchedulerEstimateDecodeWorkNs(
        api->scheduler,
        mtp_request_count,
        SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        api->decode_execution_row_capacity,
        &mtp_work_ns);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    /* Compare plain decode against the full MTP cycle: the verify batch
     * plus the draft-chain dispatches that produced the candidates.
     * The measured-plan model cannot see the serialized draft work, so
     * scale the plain estimate by the calibrated chain multiplier. */
    if (plain_work_ns > UINT64_MAX /
            SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER)
    {
        return 0u;
    }
    {
        uint64_t draft_chain_work_ns;
        uint64_t mtp_cycle_work_ns;

        draft_chain_work_ns =
            plain_work_ns *
            SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER;
        if (mtp_work_ns > UINT64_MAX - draft_chain_work_ns)
        {
            return 0u;
        }
        mtp_cycle_work_ns = mtp_work_ns + draft_chain_work_ns;
        mtp_expected_tokens_scaled =
            SparkRequestModelMtpExpectedCommittedTokensScaled(api);
        return mtp_expected_tokens_scaled * mtp_request_count * plain_work_ns >
            SPARK_GLM52_REQUEST_API_MTP_UTILITY_MARGIN_SCALE *
            plain_request_count * mtp_cycle_work_ns;
    }
}




uint32_t SparkRequestModelSlotCanSpeculate(
    const SparkRequestApi *api,
    const SparkRequestApiSlot *slot)
{
    if (!SparkRequestModelSpeculationIsEnabled(api) || slot == 0)
    {
        return 0u;
    }
    if ((slot->flags &
            SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u)
    {
        return 0u;
    }
    if ((SparkRequestModelSpec(api)->policy_flags &
            SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_REALTIME) != 0u &&
        SparkRequestApiSlotHasRealtimePriority(slot))
    {
        return 1u;
    }
    if ((SparkRequestModelSpec(api)->policy_flags &
            SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE) != 0u)
    {
        return 1u;
    }
    return 0u;
}

uint32_t SparkRequestModelSpeculatorIsValid(const struct SparkRequestModelSpeculator *speculator)
{
	return speculator != 0 &&
		SparkGlm52DsparkValidate((const SparkGlm52DsparkSpeculator *)speculator) == SPARK_STATUS_OK;
}

uint64_t SparkRequestApiMtpResolvedRequestCount(const SparkRequestApi *api, uint64_t *committed_token_count_out);
uint32_t SparkRequestApiSlotHasRealtimePriority(const SparkRequestApiSlot *slot);
uint32_t SparkRequestApiSlotRemainingDecodeBudget(const SparkRequestApiSlot *slot);

SparkStatus SparkRequestModelPrepareDraftForSlot(
    SparkRequestApi *api,
    SparkRequestApiSlot *slot)
{
    SparkGlm52DsparkDraftRequest draft_request;
    uint64_t tap_generation;
    uint32_t requested_token_count;
    SparkStatus status;

    if (!SparkRequestModelSlotCanSpeculate(api, slot) ||
        SparkRequestApiSlotRemainingDecodeBudget(slot) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52DsparkMarkVerifierTapsReady(
        SparkRequestModelSpec(api),
        slot->request_id,
        slot->sequence_id,
        (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count,
        &tap_generation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    requested_token_count = SparkRequestModelSpec(api)->default_speculative_token_count;
    if (requested_token_count + 1u > SparkRequestApiSlotRemainingDecodeBudget(slot))
    {
        requested_token_count =
            SparkRequestApiSlotRemainingDecodeBudget(slot) - 1u;
    }
    if (requested_token_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(&draft_request, 0, sizeof(draft_request));
    draft_request.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    draft_request.descriptor_bytes =
        SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
    draft_request.requested_token_count = requested_token_count;
    draft_request.priority = slot->priority;
    draft_request.request_id = slot->request_id;
    draft_request.sequence_id = slot->sequence_id;
    draft_request.sequence_position =
        (uint64_t)slot->computed_prompt_token_count +
        (uint64_t)slot->completed_decode_token_count;
    draft_request.tap_generation = tap_generation;

    status = SparkGlm52DsparkEnsureDraft(
        SparkRequestModelSpec(api),
        &draft_request);
    if (status == SPARK_STATUS_OK)
    {
        slot->state =
            SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
        api->dspark_draft_ready_count += 1u;
    }
    return status;
}

// Accessor tier: the schedulers' remaining touches, one cast away.
SparkStatus SparkRequestModelGetDraft(SparkRequestApi *api, uint64_t sequence_id, SparkRequestModelDraftResult *result)
{
	return SparkGlm52DsparkGetDraft(SparkRequestModelSpec(api), sequence_id, result);
}

SparkStatus SparkRequestModelMarkVerifierTapsReady(SparkRequestApi *api, uint64_t request_id, uint64_t sequence_id, uint64_t sequence_position, uint64_t *tap_generation_out)
{
	return SparkGlm52DsparkMarkVerifierTapsReady(SparkRequestModelSpec(api), request_id, sequence_id, sequence_position, tap_generation_out);
}

uint32_t SparkRequestModelDefaultSpeculativeTokenCount(const SparkRequestApi *api)
{
	return SparkRequestModelSpec(api)->default_speculative_token_count;
}



SparkStatus SparkRequestModelCompleteVerify(SparkRequestApi *api, uint64_t sequence_id, const SparkRequestModelVerifyResult *verify_result)
{
	return SparkGlm52DsparkCompleteVerify(SparkRequestModelSpec(api), sequence_id, verify_result);
}

SparkStatus SparkRequestModelResolveVerifierTokens(const uint32_t *draft_token_ids, uint32_t draft_token_count, const uint32_t *verifier_token_ids, uint32_t verifier_token_count, SparkRequestModelVerifyResult *verify_result)
{
	return SparkGlm52DsparkResolveVerifierTokens(draft_token_ids, draft_token_count, verifier_token_ids, verifier_token_count, verify_result);
}

SparkStatus SparkRequestModelCancelSequence(SparkRequestApi *api, uint64_t sequence_id)
{
	return SparkGlm52DsparkCancelSequence(SparkRequestModelSpec(api), sequence_id);
}
