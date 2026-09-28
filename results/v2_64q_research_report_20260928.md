# Hybrid Routing v2.0.0: 64q confirmation and line-ordering ablation

## Question and setup

Does the frozen Hybrid router still beat a QAOA-specialized Qiskit pipeline at 64 logical qubits, and how much does Rust's optimized initial line order contribute?

We compiled MaxCut/QAOA circuits with 64 logical qubits, medium50 or dense75 interaction graphs, and one or two QAOA layers. Every logical circuit was compiled on the full FakeBrisbane, FakeKyiv, and FakeSherbrooke targets. The main comparator was Qiskit `Commuting2qGateRouter` with SAT first-layer placement and a fixed 2-second limit **per feasibility check**. Stock Qiskit Level 3 was a secondary reference. The Qiskit seed, graph generator, time limits, and routing code were unchanged between the first holdout and the independent confirmation. These are local compiler tests, not QPU measurements.

The first holdout used graph IDs 100700–100704. Confirmation used **different IDs, 101000–101004**. Each run comprised 20 distinct logical circuits compiled on three targets: 60 paired target cases. The [research bundle](v2_64q_final_bundle_20260928.zip) contains the frozen source and original data. Source hashes were checked again after confirmation and remained unchanged.

## Independent confirmation

| Run | Paired cases | Completed Hybrid routes | Hybrid native 2Q | Commuting + SAT native 2Q | Hybrid change | Hybrid wins / ties / losses | Hybrid / commuting 2Q depth | Mean compile time, Hybrid / commuting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First holdout | 60 | 60 | 509,815 | 535,260 | **−4.8%** | 60 / 0 / 0 | 16,896 / 17,352 | 3.00 / 10.96 s |
| Independent confirmation | 60 | 60 | 510,530 | 535,392 | **−4.6%** | 60 / 0 / 0 | 16,944 / 17,328 | 3.00 / 10.81 s |
| **Both runs** | **120** | **120** | **1,020,345** | **1,070,652** | **−4.7%** | **120 / 0 / 0** | **33,840 / 34,680** | **3.00 / 10.88 s** |

Across both runs, Hybrid used **2.4% less summed 2Q depth** and was **3.6× faster on mean compilation time** than this specialized pipeline. It had lower depth in 105/120 target cases and lower compile time in 120/120. The 120 target cases represent **40 distinct logical circuits**, because each circuit was repeated on three targets. Peak process RAM was 952 MiB. Stock Level 3 used 1,526,391 native 2Q gates across these same 120 cases; Hybrid used 33.2% fewer.

Both graph families repeated the advantage: medium50 used 499,752 versus 533,580 native 2Q gates (**−6.3%**); dense75 used 520,593 versus 537,072 (**−3.1%**). Every selected Hybrid route in both runs was the Rust optimized-line candidate. The fixed SAT limit did not produce a proof of optimal first-layer placement in **any of the 120 cases**. The specialized Qiskit pipeline did finish every case and supplied a feasible mapping. These results establish a win against that **time-limited pipeline**, not against every possible Qiskit configuration or an optimal SAT solution.

## What the ordering search contributes

The ablation used another disjoint graph range, **101100–101104**: 20 logical circuits, again compiled on the three targets. It compares one plain line route with one line route using the 0.15-second Rust initial-order search. Both use the same physical line, Qiskit seed, and native compiler. Neither gets a candidate tournament or beam search; optimized compile time includes the order-search time.

| 64q line route | Native 2Q | Summed 2Q depth | Mean compile time |
| --- | ---: | ---: | ---: |
| Plain logical order | 535,968 | 18,228 | 1.26 s |
| Rust optimized order | **512,799** | **16,929** | 1.40 s |

The optimized order won **60/60** paired target cases by native 2Q count. It used **4.3% fewer 2Q gates** and **7.1% less 2Q depth**, while adding about **0.14 s** per circuit. It won all 30 medium50 cases (−5.2% 2Q) and all 30 dense75 cases (−3.4% 2Q). Peak process RAM was 500 MiB. This isolates a real benefit from the ordering search. Because the ablation uses different circuits and deliberately omits the tournament, its 4.3% should not be interpreted as an exact decomposition of the 4.7% margin against Qiskit.

## Validation and limits

The two holdouts passed 240 candidate-route symbolic replays, 120/120 selected compiled final-mapping checks, and target-native 2Q support checks for every circuit. The ablation passed symbolic SWAP/gate-stream replay, mapping, coupling-edge, and native-support checks for both variants in all 60 cases. No 64q statevector or full compiled-output quantum equivalence test was run; a 64q statevector is unsuitable for this 16 GB machine. No circuit in this report ran on a real QPU, so hardware fidelity is unknown. Reported compile times include the benchmark's routing checks, not just a production compiler call.

The earlier [32q paired study](v2_32q_combined_20260928.md) found a 7.2% 2Q advantage on 120 cases. [40q and 48q tests](v2_40_48_64_smoke_20260928.md) are still smoke tests and four-case paired probes, so they do not establish a reliable win rate at those sizes. Historical 16–24q studies used different graph sets and successive code revisions; their percentages should not be treated as points on one controlled size-scaling curve.

**Conclusion:** On two fresh 64q MaxCut/QAOA series using these three IBM fake targets and this fixed Qiskit commuting + SAT setup, frozen Hybrid built every route and used fewer native 2Q gates in every paired target case. An independent ablation shows that optimized line ordering is a substantial part of the result. The next external claim should state the workload, target snapshots, SAT limit, and symbolic-only correctness scope explicitly.

Data: [first holdout CSV](v2_64q_fresh_holdout.csv) · [confirmation CSV](v2_64q_independent_confirmation.csv) · [ablation CSV](v2_64q_line_order_ablation_20260928.csv). The [bundle](v2_64q_final_bundle_20260928.zip) also includes the original per-run reports.
