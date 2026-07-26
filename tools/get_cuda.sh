#!/bin/sh
# Fetch nvcc for a container that has none.
#
# A previous handoff recorded that nvcc was unavailable here and that
# pip install nvidia-cuda-nvcc-cu12 ships ptxas only. The first half is wrong:
# the pip wheel is indeed ptxas alone, but NVIDIA's redistributable tarballs
# contain nvcc, cicc and libnvvm and install with no driver and no root.
#
# That mattered more than it sounds. Compiling for real immediately found two
# errors nothing else could: -arch=sm_121a silently emits .target sm_121 and
# drops every architecture-specific instruction, and the static __shared__ limit
# is 48 KB rather than the 128 KB of L1/shared the SM has.
set -e
PREFIX=${1:-/opt/cuda}
BASE=https://developer.download.nvidia.com/compute/cuda/redist
WORK=$(mktemp -d)
curl -sL "$BASE/redistrib_12.9.0.json" -o "$WORK/r.json"
for component in cuda_nvcc cuda_cudart cuda_cccl cuda_crt
do
	path=$(python3 -c "
import json,sys
d=json.load(open('$WORK/r.json'))
c=d.get('$component')
print(c['linux-x86_64']['relative_path'] if c and 'linux-x86_64' in c else '')")
	[ -z "$path" ] && continue
	file=$(basename "$path")
	curl -sL "$BASE/$path" -o "$WORK/$file"
	tar xf "$WORK/$file" -C "$WORK"
done
mkdir -p "$PREFIX"
for dir in "$WORK"/*-archive
do
	[ -d "$dir" ] && cp -rn "$dir"/* "$PREFIX"/ 2>/dev/null || true
done
rm -rf "$WORK"
"$PREFIX/bin/nvcc" --version | tail -1
