# Enhanced coarse workload generation

The original coarse compiler accounts mainly for weights and KV. The opt-in
analytic policy adds activation and intermediate tensor footprints derived from
model structure and forward shapes. HBFSim still models device service.
Neither path executes inference or requires GPU capture at runtime.

## Standalone usage

Install from this checkout with `python -m pip install -e .`. Convert a supported
Hugging Face config without loading model weights:

```sh
python -m hbserve model /path/to/config.json --hf-config \
  --dtype bfloat16 --output /path/to/model.json
python -m hbserve run \
  --model /path/to/model.json --requests /path/to/requests.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy dense-16bit-swiglu --out /path/to/new-output
```

Use `all-hbm` for the other placement. Requests must match the model ID.
The simulator and configuration are explicit inputs; there is no bundled
historical executable or dependency on a research workspace.

To examine a finite-cache assumption, add:

```sh
--coarse-l2-capacity-bytes 41943040 \
--coarse-l2-write-policy write-allocate-fetch
```

This is 40MiB (41943040 bytes). An alternative policy,
`full-store-no-fetch`, omits read fill on complete sector overwrites; partial
stores still fetch. Omit cache options for the recommended fast baseline.
Omit all coverage options to keep the original compiler/default behavior.

## Dense16 policy scope

Uniform dense BF16/FP16 SwiGLU/RMSNorm models with explicit hidden/intermediate
sizes, query/KV heads, head and rotary dimensions. B1 unchunked prefill from
context zero, followed by single-token output-emitting decode forwards;
`memory_only`. Analytic coverage rejects unsupported quantization, MoE,
sliding-window attention, batching/chunking and mixed KV precision.
A model structure does not identify framework fusion: fused attention, RoPE,
residual normalization and last-token logits are declared assumptions.
Instruction repeats, internal attention/GEMM scratch and compute are omitted.

## Cache boundary

The cache is a cold, fully associative 32B-sector interval LRU, write-back and
write-allocate, over all application objects before placement. Weight/KV state
persists across forwards. Temporary workspace is flushed/rebound each forward,
and the final forward drains dirty data. Controller metadata/NAND traffic is
outside this cache. It does not model GPU sets, L1, warp issue order or L2 hit
latency, and is not calibrated to NVIDIA hardware. Input and result receipts
record assumptions, fills, writebacks, hit/miss counts and interval/state peaks.

The included input-bound captured profile is a separate fixed-input diagnostic;
it is not required for analytic generation. See [technical details](coarse-coverage.md)
and [included evidence](coarse-evidence.md). Coarse approximation supports
fast device studies; hardware traffic and memory-only simulated time must not
be presented as validated inference latency.

## Catalog-wide model-derived policy

`--coarse-coverage-policy model-derived` covers all five packaged public descriptors:
Llama3.1-8B W8, Llama3.1-70B W8A16, Qwen3-8B BF16, Qwen3-235B FP8 MoE/GQA,
and DeepSeek-V3 FP8 MoE/MLA. The original weight, quantization-scale and KV
ledgers remain unchanged. The analytic provider adds external16-bit tensor
footprints, routed token dispatch/expert activation/combine and shared-expert
accesses, plus low-rank query/KV and compressed-context MLA accesses.

For a public descriptor use the same run command, replacing the policy value:

```sh
python -m hbserve run --model models/deepseek-v3-fp8-kv-bf16.json \
  --requests /path/to/requests.json --router examples/ci-synthetic-router.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy model-derived --out /path/to/new-output
```

MoE still requires the original router trace/provider. Its selected-expert
weight reads are preserved; this rule does not manufacture model routing.
Use the same cache options as the dense provider when desired. Device capacity
and runtime workspace must fit; supporting a descriptor does not mean it fits
an arbitrary system configuration.

To convert a public descriptor into portable native JSON with coverage geometry:

```sh
python -m hbserve model models/llama31-8b-w8-kv-bf16.json \
  --coverage-descriptor --output /path/to/model.json
```

Without this flag, conversion/default run retains its old canonical data/digest.
The optional embedded descriptor is checked against weight/KV/FLOP ledgers;
older strict readers do not recognize the new optional field. Native ledgers
without explicit dimensions cannot reconstruct quantized/MoE architecture.
Existing supported HF-imported dense structures also work with model-derived;
the established dense16 policy and its numerical results remain available.

These are declared footprint assumptions: external activations are16-bit;
quantized GEMMs consume weights/scales directly without an expanded weight
workspace. Activation re-quantization scratch is omitted. MLA uses fused latent
attention without external expanded-context K/V; MoE uses packed top-k copies
without expert-capacity padding, token drops or network traffic. Matrix weight
dtype alone does not identify a backend's activation/fusion implementation.
These assumptions are recorded in the generated profile, not presented as
hardware-qualified kernel traffic. Scope remains B1 full prefill/single-token
decode, memory_only, one model; scheduler/batch and compute support are unchanged.
