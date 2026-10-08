# Configuration provenance

The checked-in system configurations are exploratory simulator profiles, not
vendor performance guarantees. They are included so users can exercise
placement and protocol behavior with a compatible HBFSim build.

The system profiles and bundled `hbfsim_client` were synchronized with HBFSim
on 2026-09-17. They use its current OCP HBF v0.7.0 / HBM4 configuration and
simulation-session contracts. Removed subarray,
separate program-verification, arbitrary HBIO-bandwidth, page-run acceleration
and read-buffer options are not accepted by the current engine. Use the current
profiles with a current HBFSim binary; do not reuse those removed options as
compatibility overlays. Original frozen runs retain their original profiles.

`4hbm-4hbf.cfg` and its `-miniquick` variant combine a four-stack HBM domain
with four HBF stacks. Link rates, queue depths, controller timing, flash-media
timing, overprovisioning, and thermal parameters are modeling assumptions. The
mini profile reduces capacity and host work for fast integration tests; it is
not a smaller hardware product claim.

`eight-stack-baseline.cfg`, `simulation-session-mini.cfg`, `cxl-memory.cfg`,
and `nvme-ssd.cfg` exercise the baseline HBM domain, persistent session
protocol, and external backing alternatives.

`windows/` contains matched miniquick/full-scale experiments. Its paths resolve
relative to each experiment JSON. `systems/` and `overlays/` provide their full
topology matrix, capacity controls, and thermal assumptions. The original flat
profiles remain small request-serving examples; they are not alternate workload
generators. See [Fixed windows](windows.md) for scale and completion boundaries.

The `overlays/backing/calibrated/dana-a100-*` coefficients are inherited from
HBFSim's A100 host-offload fits, not portable DRAM/NVMe specifications. The
upstream raw measurements and validation receipts are not shipped here. This
standalone package therefore treats them as example parameterizations, not
self-contained calibration evidence; validate or replace them for your setup.

For publishable results, freeze the complete config set, hash it in the run
receipt, cite a source or calibration artifact for every physical parameter,
and validate against a holdout workload. Parameters without such evidence must
remain labeled assumptions or sensitivity variables.

## Calibrated GPU operators

A run config may select `timing.type = "gpu_calibrated"`, with an absolute
`profile` JSON path, `model_bindings` (model ID to `8b`, `70b`, or `235b`) and
`prefetch_depth: 0`; set `compute` to null. The CLI binds the packed weight
layout before placement and uses the profile's 256-token KV blocks. Supply
sufficient HBM runtime scratch. The provider reports the profile hash and
measured validation errors in the result; `calibrated` does not imply a
production SLO guarantee.

The corresponding profile/evidence lives in the sibling HBFSim repository at
`evidence/hardware/gpu_operators/`. The v3 runtime rejects old contiguous-KV v2
profiles. Its native client requires dynamic memory-span barrier support.

Scheduler semantics match the tested vLLM 0.26.0 synchronous FCFS configuration:
running requests (including unfinished prefills) in admission order, then
waiting requests; incremental KV allocation; complete prefix blocks shareable
within a batch; youngest-running-request preemption. Prefix release frees
suffixes first, and active entries remain protected during capacity eviction.
This does not claim vLLM async scheduling, speculative decoding, full-prompt
reservation, TP/EP or every scheduler version.

## Backend compatibility and OCP migration

Current compatibility target: HBFSim **e9ddd1c8980d56f9f6a14e5241fb7903b6b6636c**
(public release, 2026-09-30). Profiles and the bundled client are copied from
this exact source. `configs/ocp-profile-source.json` records all 13 profile
hashes and parameter changes; `previous_migrations` preserves the earlier OCP
migration receipt. `configs/backend-source.json` records client file hashes.
`configs/parameter-provenance.json` retains upstream evidence and assumptions.
These source identities do not establish hardware calibration.

The public backend uses the HBM4 channel-aggregate-v2 model. Removed DRAM
command/row/refresh parameters must not be carried into these profiles. Its
HBM service latency, scheduling quantum and bandwidth efficiency replace the
old model, so migrated experiments need new configuration identities and
must not inherit old fits or reported hardware errors. Miniquick and full
profiles differ in capacity only. Zero-HBM physical execution remains
unsupported by the controller-HBM contract. Window preflight reports this
execution limit; select supported rows explicitly with `--topologies`, for
example `all-hbm,6h2f,4h4f,2h6f,8h0f-dram,8h0f-ssd,8h0f-cxl-ssd`.

The current transaction client has fixed-duration dependency barriers and
no `span_start` or `span_scale` wire fields. Placement emits the current
transaction representation for `roofline`, `memory_only` and `linear` timing.
A legacy semantic span with unit scale is a dependency barrier; non-unit
scales are rejected before placement mutation. Existing GPU calibration
profiles that scale observed memory spans require a new execution design and
revalidation; deleting their scale would change compute/memory overlap.
GPU calibration evidence and optional GPU runtimes are not supplied by this
migration. A successful synthetic run establishes software wiring only.

Both entry points are supported: ServeLoop's bundled client and HBFSim's
source client first on `PYTHONPATH`. The native session receipts include the
current HBM resolution, controller reservations, transaction census, separate
HOST_DRAM/external accounting, zone-managed image validation, wear v2 and
checkpoint lifecycle. No older receipt fallback is introduced.

Run focused checks with an explicit backend:

```sh
python3 -B tests/test_config_compatibility.py \
  --hbfsim-root /path/to/HBFSim --simulator /path/to/HBFSim/build/hbfsim \
  --client bundled
python3 -B tests/test_config_compatibility.py \
  --hbfsim-root /path/to/HBFSim --simulator /path/to/HBFSim/build/hbfsim \
  --client external
python3 -B tests/test_hbserve.py --simulator /path/to/HBFSim/build/hbfsim
python3 -B tests/test_session_compatibility.py --simulator /path/to/HBFSim/build/hbfsim
python3 -B tests/test_windows.py --simulator /path/to/HBFSim/build/hbfsim
```

Native tests cover bounded serving/window executions, configuration acceptance,
conservation and session lifecycle. They do not execute every full-scale
physical matrix or validate absolute inference latency. Historical experiment
inputs and results retain their original scope; freeze new inputs for reruns.


## MoE prefix routing identity

Python integrations may enable MoE prefix caching by passing the same router to
`HBServeCompiler` and `HBServePlacement`. The router must implement
`PrefixRouterProvider.prefix_block_keys(request=..., model=..., block_tokens=...)`.
Return one key for each complete prompt block. Equal keys must imply the same
model, causal token prefix and expert decisions for all layers through that
block. Include the parent chain, model, cache salt and block geometry.

A router determined solely by causal token identity may namespace content keys
with its algorithm and parameters. A recorded or request-indexed router must
also include the relevant expert decisions; identical tokens alone are
insufficient. Providers without this contract still cannot enable MoE prefix
caching. Placement checks that each compiled batch uses the same router digest
and records `prefix_router_sha256` in its receipt.
