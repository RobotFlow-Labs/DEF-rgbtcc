# Tasks — DEF-rgbtcc

## Task Board
| ID | Title | Priority | Depends On | Status |
|---|---|---|---|---|
| PRD-001 | Environment and package baseline (py3.11/uv) | P0 | - | TODO |
| PRD-002 | Dataset + checkpoint contract validation | P0 | PRD-001 | TODO |
| PRD-003 | Training command pipeline bootstrap | P0 | PRD-001, PRD-002 | TODO |
| PRD-004 | Evaluation and metrics export path | P0 | PRD-003 | TODO |
| PRD-005 | Benchmark suite baseline | P1 | PRD-001 | TODO |
| PRD-006 | CUDA profiling and hotspot lock | P1 | PRD-005 | TODO |
| PRD-007 | CUDA kernel integration phase | P1 | PRD-006 | TODO |
| PRD-008 | MLX parity phase | P2 | PRD-006 | TODO |
| PRD-009 | Dual-compute validation report | P1 | PRD-007, PRD-008 | TODO |
| PRD-010 | Deployment/integration handoff | P0 | PRD-009 | TODO |

## Critical Path
`PRD-001 -> PRD-002 -> PRD-003 -> PRD-004 -> PRD-005 -> PRD-006 -> PRD-007 -> PRD-009 -> PRD-010`

## Fast Start
1. Complete `PRD-001` and `PRD-002` on CUDA server.
2. Run `PRD-005` benchmarks before any kernel work.
3. Lock optimization order in `PRD-006` before coding kernels.
