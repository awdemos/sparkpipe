// The multi-model link proof: a binary that touches the request tier and
// the model seam without a single glm object. Links the null provider;
// if this binary exists, any drafter-less family can host the request
// api. The assertions exercise the fail-closed contract.
#include <assert.h>
#include <stdint.h>
#include "sparkpipe/spark_request_model.h"

int main(void)
{
	assert(SparkRequestModelSpeculationIsEnabled(0) == 0u);
	assert(SparkRequestModelSpeculatorIsValid(0) == 1u);
	assert(SparkRequestModelSlotCanSpeculate(0, 0) == 0u);
	assert(SparkRequestModelCancelSequence(0, 7u) == SPARK_STATUS_OK);
	assert(SparkRequestModelDefaultSpeculativeTokenCount(0) == 0u);
	assert(SparkRequestModelMtpOutranksPlainDecode(0, 3u, 3u) == 0u);
	return 0;
}
