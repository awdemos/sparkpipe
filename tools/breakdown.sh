#!/bin/sh
# Complete code breakdown by module. Code only - tests, docs and diagnostics are
# counted separately by tools/metric.sh because they are not the denominator.
cd "$(dirname "$0")/.." || exit 1
total=0
for d in inference/kernels inference/llms inference/stage api scheduler cache ring runtime \
         model-families modules tools deployment include src core validation
do
	[ -d "$d" ] || continue
	n=$(git ls-files "$d" 2>/dev/null | grep -cE '\.(c|cu|cuh|h|py|sh)$')
	[ "${n:-0}" -eq 0 ] && continue
	l=$(git ls-files "$d" | grep -E '\.(c|cu|cuh|h|py|sh)$' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
	printf "  %-22s %4s files %8s lines\n" "$d" "$n" "${l:-0}"
	total=$((total + ${l:-0}))
done
printf "  %-22s %4s       %8s lines\n" "TOTAL" "" "$total"
