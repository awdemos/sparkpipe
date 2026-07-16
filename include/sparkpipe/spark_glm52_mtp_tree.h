#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#define SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT 5u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT 6u
#define SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT 3u
#define SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT 4u
#define SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION 3u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX 2u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX 3u
#define SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX 4u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW 0u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW 1u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW 2u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW 3u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW 4u
#define SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW 5u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT 4u
#define SPARK_GLM52_MODEL_MTP_TREE_ANCESTOR_COPY_COUNT 6u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_POSITION_COUNT 3u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH1_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH2_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH3_INDEX 2u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_PRIMARY_INDEX 0u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX 1u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_PRIMARY_INDEX 2u
#define SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX 3u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE 0u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1 1u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY 2u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE 3u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY 4u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE 5u
#define SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT 6u

typedef struct SparkGlm52MtpTreeResolution
{
	uint32_t path_id;
	uint32_t accepted_token_count;
	uint32_t committed_token_count;
	uint32_t fallback_row_index;
} SparkGlm52MtpTreeResolution;

static inline uint32_t SparkGlm52MtpTreeVerifierPositionOffset(
	uint32_t row_index)
{
	if (row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW)
		return 0u;
	if (row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW)
		return 1u;
	if (row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW ||
		row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW)
		return 2u;
	if (row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW ||
		row_index == SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW)
		return 3u;
	return UINT32_MAX;
}

static inline uint32_t SparkGlm52MtpTreeAcceptedTokenCount(uint32_t path_id)
{
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1)
		return 1u;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY ||
		path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return 2u;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY ||
		path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return 3u;
	return 0u;
}

static inline uint32_t SparkGlm52MtpTreeFallbackRowIndex(uint32_t path_id)
{
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY)
		return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW;
	return SparkGlm52MtpTreeAcceptedTokenCount(path_id);
}

static inline uint32_t SparkGlm52MtpTreeTailCandidateIndex(uint32_t path_id)
{
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY)
		return SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY)
		return SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX;
	return SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX;
}

static inline uint32_t SparkGlm52MtpTreeTailParentRowIndex(uint32_t path_id)
{
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY ||
		path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW;
	if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY ||
		path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW;
	return SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW;
}

static inline uint32_t SparkGlm52MtpTreeTailBasePositionOffset(
	uint32_t path_id)
{
	uint32_t accepted_token_count;
	accepted_token_count = SparkGlm52MtpTreeAcceptedTokenCount(path_id);
	return accepted_token_count == 0u ? 0u : accepted_token_count - 1u;
}

static inline uint32_t SparkGlm52MtpTreeResolutionIsValid(
	uint32_t proposed_token_count,
	uint32_t accepted_token_count,
	uint32_t path_id)
{
	if (accepted_token_count > proposed_token_count)
		return 0u;
	if (proposed_token_count == 0u)
		return accepted_token_count == 0u &&
			path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (proposed_token_count != SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (path_id >= SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT)
		return 0u;
	return accepted_token_count ==
		SparkGlm52MtpTreeAcceptedTokenCount(path_id);
}

static inline SparkStatus SparkGlm52MtpTreeResolve(
	const uint32_t *candidate_token_ids,
	const uint32_t *verifier_token_ids,
	SparkGlm52MtpTreeResolution *resolution)
{
	uint32_t path_id,token_index;
	if (candidate_token_ids == 0 || verifier_token_ids == 0 ||
		resolution == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (token_index=0u;
		 token_index<SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
		 token_index++)
	{
		if (candidate_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	for (token_index=0u;
		 token_index<SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
		 token_index++)
	{
		if (verifier_token_ids[token_index] >=
			SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	if (verifier_token_ids[
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW] ==
		candidate_token_ids[SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX])
	{
		path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1;
		if (verifier_token_ids[
				SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] ==
			candidate_token_ids[SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX])
		{
			path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY;
			if (verifier_token_ids[
					SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] ==
				candidate_token_ids[
					SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX])
				path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY;
			else if (verifier_token_ids[
					SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] ==
				candidate_token_ids[
					SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX])
				path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE;
		}
		else if (verifier_token_ids[
				SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] ==
			candidate_token_ids[SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX])
			path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE;
	}
	resolution->path_id = path_id;
	resolution->accepted_token_count =
		SparkGlm52MtpTreeAcceptedTokenCount(path_id);
	resolution->committed_token_count = resolution->accepted_token_count + 1u;
	resolution->fallback_row_index =
		SparkGlm52MtpTreeFallbackRowIndex(path_id);
	return SPARK_STATUS_OK;
}
