# Operator coverage for the fast coarse path

`hbserve run --coarse-coverage-profile FILE` adds activation/control/scratch
coverage to the original weight/KV compiler. The original path remains the
default when the option is absent. GPU compute calibration is not required;
the current coverage interface requires `--timing memory_only`.

The packaged example supports exactly the included Qwen2.5-1.5B BF16/SGLang
model and request digests: B1,P16 plus one actual decode at context17. It uses
previously qualified tensor/kernel coverage, not full observation files or
individual GPU requests. Different models, input traces, batch shapes and
contexts are rejected instead of silently extrapolated. This is a bounded
reusable profile interface, not a general model/shape generator.

From the HBF research workspace, use a fresh output directory:

```sh
python tools/hbf.py hbserve run \
  --model ServeLoop/examples/coarse-coverage/model.json \
  --requests ServeLoop/examples/coarse-coverage/requests.json \
  --system /path/to/system.cfg --simulator /path/to/hbfsim \
  --placement weights-hbf-kv-hbm --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-profile ServeLoop/examples/coarse-coverage/qwen2.5-1.5b-sglang-bf16-p16-d1.json \
  --out /path/to/fresh-output
```

Use `--placement all-hbm` for the other qualified placement. No experimental
environment overrides are required. A standalone installation uses the same
arguments with `python -m hbserve run`. The HBFSim binary/configuration are
explicit inputs; the profile does not pick a historical backend automatically.

## Meaning and boundaries

- Original static weight ranges/KV writes are retained. Decode KV reads include
  the newly appended token; prefill attention reads dense K/V temporaries.
- Norm, QKV, RoPE, residual, MLP and tail activations use tensor footprints.
  RoPE reads indexed position rows, not the entire allocated lookup table.
- Attention uses per-kernel unions from existing captured pre-cache sectors,
  including touched copies/scratch and explicitly modeled SASS local frames.
  Repeated instructions are collapsed; this is not measured GPU post-L2 traffic.
- Object-relative workspace identities/offsets are retained, with disjoint
  packed placement in the HBM runtime reservation. It is not a reconstruction
  of GPU physical placement. Profile ranges must fit that reservation.
- Zero-duration completion joins retain dependencies without a cross-product
  of edges. Original coarse weight/KV prefetch remains; nonattention operations
  are not a fully matched GPU kernel issue schedule.
- In the included captured window, omitted allocator buffers have0 compiled
  requests except592B/23 pre-cache requests in prefill cumsum. This omission is
  quantified for this input only, not assumed valid for other kernels/shapes.

The profile and its byte digest are recorded in the run inputs. The CLI labels
the traffic as coverage and reports that GPU cache misses are not modeled.
Model/request canonical digests and shape checks prevent accidental reuse.

## Cache decision

Default `--coarse-cache-bound off` retains the coverage approximation. Optional
`--coarse-cache-bound ideal-temporaries` removes only temporary coverage traffic,
while retaining weight/KV requests and dependencies. This is a sensitivity
bound assuming temporary data stays on-chip, not a finite GPU cache model.
It cannot bound effects of weight/KV caching, repeats or real GPU scheduling.

For the included two placements, the bound changes total simulated memory time
by at most~3.25% and preserves their ranking. It does not justify ignoring cache
for physical traffic claims: the detailed reference's modeled cache changes
read/write volumes substantially. Do not apply a single fitted hit ratio to
coverage bytes. Use the opt-in detailed reference when post-cache traffic is
the object of study; GPU-cache accuracy is outside this coarse version's scope.

## Rebuild a profile

Runtime profiles require no observation/template binaries. An offline export
from a qualified ledger is available separately:

```sh
python -m hbserve.coarse_coverage_export \
  --ledger /path/to/qualified-ledger.json \
  --model examples/coarse-coverage/model.json \
  --requests examples/coarse-coverage/requests.json \
  --out /path/to/new-profile.json
```

The exporter currently recognizes the qualified28-layer call structure only.
Changing its input or adding a profile requires new coverage qualification;
successful schema parsing is not hardware validation. Export refuses to
overwrite an existing file. Verbose audits remain separate from compact
runtime descriptors (~219KB for this profile).

## Model-derived analytic policy (no per-input capture)

Use `--coarse-coverage-policy dense-16bit-swiglu` instead of
`--coarse-coverage-profile FILE` to generate tensor footprints from the model
ledger and request shapes. The two options are mutually exclusive. Example:

```sh
python tools/hbf.py hbserve run \
  --model ServeLoop/examples/coarse-coverage/model.json \
  --requests /path/to/requests.json --system /path/to/system.cfg \
  --simulator /path/to/hbfsim --placement weights-hbf-kv-hbm \
  --timing memory_only --prefetch-depth 1 \
  --coarse-coverage-policy dense-16bit-swiglu --out /path/to/fresh-output
```

No captured profile, observation ledger or detailed trace is read by this
provider. New models must carry explicit `structure`: hidden/intermediate
sizes, query/KV head counts, head dimension, BF16/FP16 dtype, matching KV dtype,
rotary dimension, RoPE-table dtype, SwiGLU/RMSNorm and optional Q/K head norms.
Layer count and tied embeddings remain in the model ledger. No weight-byte
reverse inference is used for structured models; incompatible weight/KV
payloads are rejected. Structure is optional in hbserve.model v2: old JSON
canonical contents/digests are unchanged. Adding structure deliberately changes
the model digest, so a captured profile bound to old JSON must still use its
original model. Older strict clients cannot read the new optional field.

Import a supported dense Llama/Qwen/Mistral Hugging Face config without loading
weights or executing the model:

```sh
python tools/hbf.py hbserve model /path/to/config.json --hf-config \
  --dtype bfloat16 --output /path/to/model.json
```

Use `--dtype float16` for FP16. KV defaults to the same dtype; mixed KV,
quantization, MoE and sliding-window attention are currently rejected for this
structure path. Eligible unquantized dense GQA public catalog descriptors also
carry structure when converted. Catalog precision is explicit and cannot be
overridden by the HF dtype flags. Existing native SGLang callers of dense_model
retain their old ledger/digest unless they explicitly request include_structure.
The old `dense-bf16-swiglu` policy name remains a compatibility alias. Legacy
unstructured1.5B JSON retains only its previously qualified geometry fallback;
other models must supply structure.

Scenario support remains B1 unchunked prefill from zero prior context and
single-token decode with positive context; every forward must emit output.
Batching, prefix/chunked prefill and non-output forwards are rejected.
Complete qualification covers1.5B,7B (BF16/FP16, P16/P128+D1) and public
Qwen3-8B (P16+D1), not arbitrary framework/hardware accuracy. Synthetic formula
tests also cover MHA/GQA, query width different from hidden width, partial
rotary dimensions and16-bit RoPE tables. Q/K head norms are represented as
separate footprint stages when declared.

The explicit execution policy assumes fused RoPE/residual-RMSNorm, fused
attention without materialized score matrices, dense prefill K/V copies and
last-token16-bit head plus FP32 logits. Selecting this
policy asserts these implementation assumptions; a model file alone cannot
identify framework fusion. Attention internal scratch/local frames, GEMM
workspace, control accesses and repeated loads are omitted. GPU caches and
compute remain unmodeled. These omissions make this a different approximation
from captured kernel-sector coverage, not an equivalent reconstruction.

Sequential layers reuse packed tensor workspace. Footprint bytes and required
HBM runtime reserve grow with token count; insufficient reserve is rejected by
placement. New activation range count stays fixed with token count, while
original embedding gathers and placement stripe splits may add operations.
`--coarse-cache-bound ideal-temporaries` is also available as the same explicit
sensitivity bound, not a finite cache model. Run receipts include the policy
and assumptions; compiled workload audit includes generated profile evidence.

P16 original and captured-profile scientific results remain unchanged. In the
complete all-HBM example analytic/captured simulated totals are0.828522/0.831579ms;
host execution0.183/0.479s (single samples). Traffic differs by about3–6%, not
hardware error. Initial shape results: research workspace
`experiments/coarse_analytic_shapes_20261008_02/RESULT.md`. Structure parameterization
and7B/8B integration results: `experiments/coarse_structure_params_20261008_03/RESULT.md`.
