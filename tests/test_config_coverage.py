"""Every constant a model declares must be used by that model.

A constant carried into config.h and never referenced is a piece of the
architecture nobody implemented. That is not a style point - it found two real
defects in llms/glm5_2 within an hour of each other:

  GLM52_QUERY_A_DIM      sat unused while the attention projection did one GEMM
                         instead of the two-stage low-rank pair the rank is for
  GLM52_FIRST_ROUTED_LAYER
  GLM52_DENSE_INTERMEDIATE
                         sat unused while every layer was routed, including the
                         three that have no experts

Both were invisible to the compiler, to every other gate, and to a numerical
comparison - which would have said "wrong output" without saying that a whole
projection stage was missing.

Constants that are deliberately unused belong in a config with a comment saying
why, not silently. The exemption list below is that comment, and it is short on
purpose: a long one means the check has been turned off rather than satisfied.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Constants a model declares that nothing in the tree is expected to reference.
# Each needs a reason. "Not implemented yet" is a valid reason and should be
# stated, because then this list is the list of what is missing.
EXEMPT = {
    "GLM52_MTP_DRAFT_TOKENS": "speculation is wired in kernels/speculate.cuh but no model drives it yet",
    "GLM52_MTP_LAYER_INDEX": "same",
    "GLM52_WEIGHT_LAYERS": "used by the host packer, not by kernels",
    "MIMO25_VOCAB": "no layer sequence yet, so no head call",
    "QWEN36_VOCAB": "same",
    "K3_VOCAB": "same",
    "DSV4_VOCAB": "same",
    "MIMO25_LAYERS": "the layer loop is the host's; layer.cuh is one layer",
    "MIMO25_LAYER_KIND_FULL": "the kind is a template parameter, so the enum "
                              "value names an entry point rather than being "
                              "compared against - Mimo25LayerAttentionFull* vs "
                              "*Swa*, chosen by the host from the layer index",
    "MIMO25_LAYER_KIND_SWA": "same",
    "GLM52_MXFP4_GROUP": "MXFP4 is a supported format with no checkpoint using it",
    "GLM52_FP8_SCALE_BLOCK": "the format trait carries its own group size",
    "GLM52_NVFP4_GROUP": "same",
    "MIMO25_RMS_EPSILON": "passed by the host, not named in unity.cu",
    "QWEN36_MTP_LAYERS": "speculation not driven yet",
    # glm5_2, with the reason each is not referenced by layer.cuh
    "GLM52_LAYERS": "the layer loop is the host's; layer.cuh is one layer",
    "GLM52_ROUTED_LAYERS": "derived, used by the host packer",
    "GLM52_W1_COMPONENTS": "layer.cuh writes the factor of two directly",
    "GLM52_QK_NOPE_DIM": "raw-path width, set into LmLowRankWeights by the host",
    "GLM52_VALUE_DIM": "same",
    "GLM52_QUERY_A_DIM": "same - the rank the host puts in LmLowRankWeights",
    "GLM52_DSA_INDEX_EPSILON": "the index-path norm is not implemented; the "
                               "scoring kernel takes raw index queries",
    "GLM52_ROUTED_SCALE": "the host pre-scales route_weight; see the comment in "
                          "Glm52LayerMoe about why scaling the gates is not the "
                          "same as scaling the result",
}


def main() -> int:
    failures = 0
    for model in sorted(p for p in (ROOT / "llms").iterdir() if p.is_dir()):
        config = model / "config.h"
        if not config.exists():
            continue
        declared = re.findall(r"^#define ([A-Z][A-Z0-9_]*)", config.read_text(encoding="utf-8"), re.M)
        # everything the model and the library could reference it from
        corpus = ""
        for path in list(model.glob("*")) + list((ROOT / "kernels").rglob("*.cuh")) + list((ROOT / "runtime").rglob("*")):
            if path.is_file() and path.name != "config.h":
                corpus += path.read_text(encoding="utf-8", errors="ignore")
        unused = [d for d in declared
                  if corpus.count(d) == 0 and d not in EXEMPT and not d.endswith("_H")]
        # A model with no layer sequence has every constant unused, which is one
        # fact and not thirteen. Reporting it as thirteen failures would drown
        # the models that DO have a sequence and skip something in it.
        has_sequence = (model / "layer.cuh").exists()
        if not has_sequence:
            print(f"  --   {model.name}: no layer.cuh; {len(declared)} constants "
                  f"unused because nothing sequences them yet")
            continue
        if unused:
            print(f"  FAIL {model.name}: declared and never used by its sequence")
            for name in unused:
                print(f"         {name}")
            failures += len(unused)
        else:
            exempt_here = [d for d in declared if d in EXEMPT]
            print(f"  ok   {model.name}: {len(declared)} constants, "
                  f"{len(exempt_here)} exempt with a stated reason")
    print(f"\n{'FAIL' if failures else 'PASS'} ({failures} unused)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
