#!/bin/sh
set -e
gcc -Iinclude -Imodel-families/glm52/include -DNDEBUG -c serving/spark_request_model_null.c -o /tmp/g_null.o
gcc -Iinclude -Imodel-families/glm52/include -DNDEBUG -c modules/glm52_dspark_draft_backend/source/spark_glm52_request_model.c -o /tmp/g_glmrm.o
nm /tmp/g_null.o | awk '$2=="T"{print $3}' | sort > /tmp/g_null.syms
nm /tmp/g_glmrm.o | awk '$2=="T"{print $3}' | grep -v MtpExpected | sort > /tmp/g_glm.syms
diff /tmp/g_null.syms /tmp/g_glm.syms
