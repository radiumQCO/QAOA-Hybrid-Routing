# Hybrid QAOA Routing v2.0.0

I built this router around a simple QAOA trick: the ZZ gates in a cost layer can be reordered. Rust searches for a good qubit order and routes the layer; Qiskit compiles the chosen circuit to the target's native gates.

On two independent 64q MaxCut/QAOA runs, Hybrid used 4.7% fewer native 2Q gates than the tested Qiskit commuting-routing pipeline and won all 120 paired cases.

## Try it in Qiskit

Install this release from GitHub (tested with Python 3.12; building from source needs Rust):

```bash
python -m pip install "git+https://github.com/radiumQCO/QAOA-Hybrid-Routing.git@v2.0.0"
```

Then use the normal Qiskit call:

```python
from qiskit import transpile

compiled = transpile(
    your_qaoa_circuit, backend=your_backend,
    optimization_level=3, routing_method="qaoa_hybrid", seed_transpiler=11,
)
print(compiled.metadata["qaoa_hybrid_route"])
```

The printout is `hybrid` when this router ran or `sabre_fallback` for other circuits or an unfinished bounded search. The plugin targets 16–64 qubit commuting-ZZ QAOA with interaction density at least 45%. Its target needs gate durations for the candidate comparison. [A complete example](examples/basic.py) uses only Qiskit. Run `python examples/basic.py` from a clone after installing the package. `python plugin_smoke.py` checks the installed plugin, native gates, measured-qubit mapping, and a 16-qubit compiled state. With [research dependencies](requirements.txt) installed, `python plugin_smoke.py --ibm` also reproduces saved FakeBrisbane 16q and 64q cases.

## Results so far

These are **local compilation results** from the research runner, not a separate benchmark of the plugin. The main comparison baseline is a QAOA-specific Qiskit pipeline: `Commuting2qGateRouter` with a time-limited SAT placement. Stock Level 3 is in the reports too.

| Workload | Paired target cases | Completed | Hybrid vs commuting native 2Q | Wins |
| --- | ---: | ---: | ---: | ---: |
| 64q, two fresh runs | 120 | 120/120 | **4.7% fewer** | 120/120 |
| 32q | 120 | 120/120 | **7.2% fewer** | 105/120 |
| 24q medium50 | 90 | 90/90 | **8.2% fewer** | 69/90 |

The 64q runs used 40 different MaxCut/QAOA circuits on FakeBrisbane, FakeKyiv, and FakeSherbrooke. Hybrid had 2.4% less summed 2Q depth and averaged 3.00 s per compile versus 10.88 s for the commuting pipeline. SAT did not prove optimal placement in those runs. [Full report](results/v2_64q_research_report_20260928.md) · [first run](results/v2_64q_fresh_holdout.csv) · [independent run](results/v2_64q_independent_confirmation.csv) · [line-order ablation](results/v2_64q_line_order_ablation_20260928.csv).

These numbers cover this workload and these three target snapshots. The 20q results were mixed, and 40/48q have only small probes. Checks replayed SWAPs and gates, verified final mapping, and checked native 2Q support. I have not fully simulated the compiled 64q circuits or run this version on a real QPU. [32q report](results/v2_32q_combined_20260928.md) · [24q report](results/v2_medium24_combined_20260928.md) · [size probes](results/v2_40_48_64_smoke_20260928.md).

The frozen benchmark code is in [research_32plus.py](research_32plus.py); the router is in [hybrid_level3_v2.py](hybrid_level3_v2.py) and [rust_core/src/lib.rs](rust_core/src/lib.rs). Licensed under [Apache-2.0](LICENSE).
