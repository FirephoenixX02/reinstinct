# AGENTS.md — reinstinct

## What this is

Custom HIP inference engine for AMD MI50/MI60 (gfx906), written in Rust.
Single crate, single binary (`reinstinct-engine`). No build.rs — HIP kernels
in `kernels/*.cpp` are embedded via `include_str!` in Rust source and
compiled on first use with `hipcc`, then cached at `~/.cache/reinstinct/kernels/`.

## Commands

```
cargo build --release          # production binary → ./target/release/reinstinct-engine
cargo test                     # unit tests (CPU oracle only, no GPU needed)
cargo test --release           # same, faster
scripts/bench-all.sh           # full decode + prefill + MTP benchmark (needs GPU + models)
scripts/bench-all.sh quick     # decode-only benchmark
```

## Kernel compilation

- Kernels are `.cpp` files in `kernels/`, `include_str!`'d into Rust source.
- Compiled at runtime by `hipcc` on first use; cached per-source-hash.
- Target arch defaults to `gfx906`. Override with `REINSTINCT_OFFLOAD_ARCH`.
- Cache key includes `hipcc` version — toolchain bump triggers recompile.
- MOE kernels in `kernels/moe_*.cpp` are shared between `gemma4.rs` and `qwen35.rs` runtimes.
- Step kernels (decode) are called with input==output aliased — never assume separate buffers.

## Tests

Most tests require a GGUF model fixture. They skip gracefully when absent.

```
cargo test                     # runs; most tests skip without fixture
REINSTINCT_GGUF_FIXTURE=/path/to/model.gguf cargo test   # override fixture path
```

Default fixture path: `~/models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`.

Golden tests (`tests/qwen35_golden.rs`) compare CPU forward against JSON fixtures
in `tests/golden/`. The golden-logits helper (`tests/golden/dump_logits`) is built
separately via `tests/golden/build.sh` against an external llama.cpp build.

GPU oracle tests use `set_dp4a(false)` + quantization-realistic tolerances + top-K comparison.

## Key source layout

```
src/main.rs          — CLI (clap), all subcommands
src/lib.rs           — public module surface
src/runtime/gemma4.rs     — Gemma 4 GPU forward (prefill + decode)
src/runtime/qwen35.rs     — Qwen 3.5/3.6 GPU forward (hybrid GDN + attention)
src/runtime/kernels.rs    — kernel compilation cache + test launchers
src/runtime/spec_decode.rs — MTP speculative decoding
src/runtime/kv_superquant.rs — tiered KV cache (int8 + turbo3)
src/hip/             — runtime dlopen of libamdhip64.so, safe wrappers
src/gguf/            — zero-copy GGUF parser (memmap2)
src/cpu/             — CPU oracle (f32 reference, validation only)
src/model/           — typed model structs (gemma4, qwen35)
src/quant/           — quantization format structs (q4_k, q5_k, q6_k, q8_0, iq4_xs, turbo3)
src/serve/           — OpenAI-compatible HTTP server
kernels/             — 122 HIP kernel source files (.cpp)
```

## Architecture

Two model families: `gemma4` (Gemma 4 dense/MoE) and `qwen35`/`qwen35moe` (Qwen 3.5/3.6).
GPU forward path uses int8 dp4a matvec + fused kernels + HIP graph capture.
CPU path is a slow f32 oracle for validation — NOT ground truth for every model
(31B dense disagrees with both GPU paths).

Zero ROCm link dependency — all HIP calls via `libloading::dlopen` of `libamdhip64.so`.

## Environment variables (high-signal subset)

Full list in `MANUAL.md`. These are the ones that change behavior agents should know about:

| Variable | Effect |
|---|---|
| `REINSTINCT_OFFLOAD_ARCH` | Override GPU arch for `hipcc` (default `gfx906`) |
| `REINSTINCT_GGUF_FIXTURE` | GGUF path for tests |
| `REINSTINCT_NO_GRAPH` | Disable HIP graph capture (needed for `REINSTINCT_DECODE_DEBUG`) |
| `REINSTINCT_PREFILL` | `generate-text` runs prefill only, then exits |
| `REINSTINCT_PREFILL_TWICE` | Two prefill passes: warm + captured measurement |
| `REINSTINCT_KV_SUPERQUANT=1` | Enable tiered KV cache (capacity feature, ~30% slower decode) |
| `RUST_LOG` | Controls tracing log level (default `info`) |

## Models

Models are NOT in the repo. Expected layout under `~/models/`:
- `~/models/gemma4-31b/` — Gemma 4 31B dense
- `~/models/gemma4-26B/` — Gemma 4 26B MoE
- `~/models/qwen-3.5-*` — Qwen 3.5 variants
- `~/models/qwen-3.6-*` — Qwen 3.6 variants
- `~/models/gemma4-mtp/` — MTP drafters

Models are Unsloth Dynamic GGUF (UD-Q4_K_XL, UD-Q6_K_XL) — per-tensor mixed quant types.

## Conventions from HANDOFF.md (preserved)

- Bench results: tok/s, never ms/tok
- `-it` models require chat-template (`--system`/`--user`), raw prompts produce garbage
- Step kernels are called input==output aliased
- `kernels/moe_*.cpp` shared between gemma4.rs and qwen35.rs
- GPU oracle tests: `set_dp4a(false)` + quant-realistic tolerances + top-K

## Docs to reference

- `MANUAL.md` — full CLI reference, all env vars, API spec, performance tables
- `docs/ARCHITECTURE.md` — engine design, gfx906 hardware constraints
- `docs/SUPERQUANT.md` — tiered KV cache design
