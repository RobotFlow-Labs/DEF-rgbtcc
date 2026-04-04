# PRD — DEF-rgbtcc (RGB-T Crowd Counting)

## 1. Document Control
- Module: `DEF-rgbtcc`
- Wave: `Wave-8 Defense`
- Date: `2026-04-04`
- Status: `Execution Ready`
- Owner: `ANIMA Module Engineering`
- References:
  - Code: `repositories/RGBT-Crowd-Counting`
  - Paper: `papers/2509.17079.pdf`
  - Module context: `CLAUDE.md`, `AGENTS.md`

## 2. Executive Summary
This module productizes the reference RGB-T crowd counting model into an ANIMA-ready package with reproducible runtime, benchmark tooling, CUDA optimization hooks, and MLX parity scaffolding. The primary output is a reliable density-map counting pipeline for surveillance and fleet safety workloads in degraded visual environments.

## 3. Problem
The reference implementation is research-grade and lacks ANIMA delivery requirements:
1. No delivery-grade project planning and slice-based execution assets.
2. No packaging boundary for integration with shared infrastructure.
3. No benchmark/validation harness for server and edge environments.
4. No kernel-integration abstraction for custom CUDA rollout.
5. No MLX parity path for Apple Silicon local verification.

## 4. Objectives
1. Produce a full engineering PRD with measurable gates.
2. Slice implementation into trackable tasks with dependencies.
3. Build production scaffolding without mutating the reference repo.
4. Prepare clean CUDA-server handoff with deployment and validation scripts.

## 5. Non-Objectives (Current Iteration)
1. Running full 400-epoch training in this local pass.
2. Completing all custom CUDA kernels in this local pass.
3. Final numerical parity signoff between CUDA and MLX.
4. Dataset mirroring/downloading in this local pass.

## 6. Scope
### In Scope
1. Planning assets (`PRD.md`, task files, next-step execution map).
2. Package scaffolding (`src/def_rgbtcc`).
3. Runtime bridge to reference model.
4. Benchmark harness (latency, throughput, memory).
5. CUDA/MLX wrapper stubs with safe fallback behavior.
6. CUDA-server deployment and validation scripts.

### Out of Scope
1. Rewriting baseline architecture from paper.
2. Changing reference training logic semantics.
3. Publishing model weights.

## 7. Users and Use Cases
1. ORACLE operator: count people under low-light/thermal-heavy scenes.
2. ATLAS fleet safety service: trigger crowd-aware navigation constraints.
3. Module engineer: benchmark and optimize inference kernels on GPU server.
4. Research engineer: compare CUDA and MLX outputs during development.

## 8. Functional Requirements
1. Provide model loader that instantiates reference `Net` from local repository path.
2. Allow optional checkpoint loading (strict/non-strict modes).
3. Provide benchmark CLIs for latency, throughput, memory.
4. Provide dataset/checkpoint validation utility.
5. Provide server deployment bootstrap scripts.
6. Keep package runtime compatible with Python `3.11` and `uv`.

## 9. Technical Requirements
1. Package manager: `uv`.
2. Python baseline: `3.11`.
3. CUDA baseline on server: PyTorch cu128 wheels.
4. No hard dependency on custom kernels for baseline path.
5. Fallback behavior required when extension import fails.
6. File-system paths configurable by flags/env.

## 10. Architecture
### 10.1 Reference Boundary
- Reference sources are read-only under `repositories/RGBT-Crowd-Counting`.
- ANIMA code imports reference model dynamically via wrapper.

### 10.2 ANIMA Package Boundary
`src/def_rgbtcc` contains:
1. `config.py` — typed runtime/train/bench configs.
2. `reference_wrapper.py` — dynamic import + model build.
3. `benchmarking/` — latency/throughput/memory runners.
4. `kernels/` — extension loader wrappers with fallback.
5. `mlx_port/` — MLX scaffolding.
6. `validation/` — dataset/checkpoint contract checks.

### 10.3 Operational Boundary
`scripts/server/` contains:
1. deploy sync script.
2. server bootstrap script.
3. runtime validation script.
4. benchmark execution script.
5. asset validation script.

## 11. Deliverables
1. Full PRD and detailed task slices.
2. Python package scaffold and importable module.
3. Benchmarks and runtime validators.
4. CUDA handoff runbook.

## 12. Success Metrics
1. Planning completeness: PRD + all task slices authored and linked.
2. Packaging completeness: module import path available and stable.
3. Operational completeness: one-command deploy/bootstrap/bench flow documented.
4. Validation completeness: dataset/checkpoint/runtime checks available.

## 13. Milestones
1. M1 Planning complete.
2. M2 Scaffold complete.
3. M3 CUDA handoff ready.
4. M4 Profiling and optimization phase start.

## 14. Risks and Mitigation
1. Missing checkpoint path.
- Mitigation: explicit validator and checkpoint parameterization.
2. Dataset layout mismatch.
- Mitigation: split/id pair validator before training.
3. CUDA extension incompatibility.
- Mitigation: fallback path preserves inference flow.
4. Server drift in Python/Torch versions.
- Mitigation: controlled bootstrap script with Python 3.11 and cu128.

## 15. Test and Validation Plan (This Stage)
1. Package import validation (`import def_rgbtcc`).
2. Reference forward smoke test.
3. Dataset contract validation script.
4. CUDA runtime smoke validation script.
5. Benchmark scripts produce deterministic formatted output.

## 16. Dependencies
1. GPU server access (`datai_srv7_development`).
2. Dataset roots:
   - `/mnt/forge-data/datasets/RGB-T-CC/RGBT-CC`
   - `/mnt/forge-data/datasets/RGB-T-CC/DroneRGBT` (optional stage)
3. Checkpoint path for reference model.

## 17. Exit Criteria for "Ready to Move to CUDA Server"
1. PRD and tasks are complete and actionable.
2. Deploy/bootstrap scripts exist and are executable.
3. Validators and benchmark entrypoints exist.
4. Next-steps document points to exact commands.

## 18. Current State
- Planning: complete.
- Scaffolding: complete for baseline package and operations.
- Training/profiling/optimization: queued in execution tasks.
