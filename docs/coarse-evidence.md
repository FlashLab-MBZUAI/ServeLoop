# Complete-model measurements and limitations

The compact [evidence capsule](../examples/coarse-coverage/evidence.json) contains
exact phase/total bytes, errors, configuration/backend identities and the two-placement
sensitivity screen. No raw traces, remote server details or research paths are
required by the runtime. This capsule preserves historical measurements; a new
candidate run is software qualification, not a new hardware acquisition.

Qwen2.5-1.5B, SGLang0.4.10, BF16, B1/TP1, RTX4000Ada. MB is decimal10^6 bytes.

## P16 + one actual decode, full model.forward

| Method | Read MB | Read error | Write MB | Write error |
|---|---:|---:|---:|---:|
| Hardware | 6153.357 | — | 98.341 | — |
| Original coarse | 6175.368 | +0.36% | 0.487 | -99.50% |
| Enhanced, no cache | 6216.716 | +1.03% | 42.361 | -56.92% |
| Enhanced,40MiB fetch | 6211.937 | +0.95% | 36.321 | -63.07% |

Activation coverage restores much of the original omitted write volume.
Finite cache modeling does not guarantee better hardware agreement.

## P512 +32 actual decode forwards, full request

The hardware window includes preparation/sampling; it is different from P16.

| Method | Read MB | Read error | Write MB | Write error |
|---|---:|---:|---:|---:|
| Hardware | 102315.986 | — | 1248.160 | — |
| Original coarse | 102370.799 | +0.05% | 15.598 | -98.75% |
| Enhanced, no cache | 103684.306 | +1.34% | 1327.208 | +6.33% |
| Enhanced,40MiB fetch | 103512.655 | +1.17% | 1133.909 | -9.15% |

The standalone traffic screen uses surrogate request ID `request`; the timing
screen uses `observed-request`. Surrogate embedding locality changes read traffic
by only9216B prefill/12288B decode, with identical writes, but neither reconstructs
GPU token addresses. Total agreement can hide phase errors:40MiB prefill read
+19.88% /write-15.92%; decode write+535.15% (hardware15.334MB,model97.392MB).

## HBFSim memory-only sensitivity

Same frozen binary/config/model/request across four sequential arms:

| Placement | No cache ms |40MiB ms | Change |
|---|---:|---:|---:|
| All HBM |13.859476|14.090134|+1.664%|
| Weights HBF, KV/temporaries HBM |97.750080|102.125739|+4.476%|

Placement ratio changes7.0529x→7.2480x (+2.766%); ordering is stable.
Prefill changes-6.304%/+6.446%, respectively. Added cache dependencies can reduce
modeled overlap even when traffic decreases. This is scoped robustness evidence
for two read-dominated placements, not validation of hardware latency, close
hybrid rankings, write-intensive SSD/CXLSSD workloads or endurance.
There are no HBF application writes here; logical payload, controller metadata
and NAND traffic remain separate. No matched detailed P512 timing is claimed.

## Catalog-wide software qualification


| Model | Original seconds | Enhanced seconds | Enhanced40MiB seconds |
|---|---:|---:|---:|
| deepseek-v3-fp8-kv-bf16.json | 10.649 | 10.808 | 11.247 |
| llama31-70b-w8a16-kv-bf16.json | 0.011 | 0.067 | 0.243 |
| llama31-8b-w8-kv-bf16.json | 0.008 | 0.032 | 0.109 |
| qwen3-235b-a22b-fp8-kv-bf16.json | 8.576 | 8.805 | 9.224 |
| qwen3-8b-bf16-kv-bf16.json | 0.007 | 0.034 | 0.119 |

Single samples, generation+cache only, not HBFSim/inference latency. MoE still runs the original per-token expert-selection provider; no per-sector expansion added. Workspace scales with tokens/top-k tensor dimensions; original weights are not expanded into BF16 allocations.

## Validation

- 21 coarse tests (including six new catalog tests) +5 structure tests passed in research and standalone candidate. Three existing optional-geometry tests passed in standalone. Both cache write policies qualified for all five models, with final dirty drain.
- All five original model digests exactly match previous standalone candidate bc4bb95 when reading the same public files. Weight objects/KV sizes and routed expert choices/weight reads are preserved. Unquantized dense profile unchanged.
- Descriptor/model mismatch,missing geometry and unsupported shapes reject. Routing-stage coverage precedes routed weight availability; byte/dependency contracts pass.
- Offline wheel build/install; all imports from installed site-packages. Nine native replay arms: actual Llama8B W8 atP16+D1, plus two small explicitly synthetic MoE/GQA andMoE/MLA fixtures, each original/enhanced/cache40. Three portable descriptor→nativeJSON conversions passed. These fixtures certify software wiring only.
- Full public671B/235B native HBFSim replay,hardware/cache fidelity and non-memory-only timing were not qualified in this task. Device capacity/CPU budgets remain explicit constraints. Earlier baseline-wide test failures remain documented; no all-CI claim.

## Compute-aware normalized roofline screen

Full five-model P512+D1/allHBM software replay now passes with original,enhanced
and enhanced40MiB modes. Capacity-only1TiB with fixed4stack timings; peak200TFLOP/s
×efficiency0.5 gives100effectiveTFLOP/s in every phase/model. This is not measured
GPU/precision calibration.

| Model | Original ms | Enhanced ms | Change | Enhanced40MiB ms |
|---|---:|---:|---:|---:|
| llama31-8b-w8-kv-bf16.json | 73.326 | 74.729 | +1.91% | 74.621 |
| llama31-70b-w8a16-kv-bf16.json | 714.215 | 721.674 | +1.04% | 724.709 |
| qwen3-8b-bf16-kv-bf16.json | 74.203 | 75.938 | +2.34% | 76.172 |
| qwen3-235b-a22b-fp8-kv-bf16.json | 250.874 | 256.302 | +2.16% | 254.423 |
| deepseek-v3-fp8-kv-bf16.json | 455.314 | 461.970 | +1.46% | 460.409 |

All modes preserve the same original FLOP/compute budget; MoE routing is counted
once. Exact20old memory_only DAG hashes reproduce unchanged. Next-layer weight/KV
prefetch overlaps aggregate compute; covered layer activations precede that compute.
This does not qualify operator-level overlap, real latency or HBF/endurance ranking.
Traffic is unchanged by enabling compute. Greater compute dominance shrinks relative
memory-model differences at this specific rate; other rates/shapes can differ.
