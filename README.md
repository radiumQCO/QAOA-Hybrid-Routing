# QAOA Hybrid Level 3 — V1

I built this because I wanted a cleaner question than “can my router win on a few nice graphs?”
The question now is:

> **If I put my QAOA-aware router in front of the normal Qiskit Level-3 pipeline and let it fall back to stock Qiskit when it should, is the whole compiler better on average?**

This folder is intentionally small. There are only two Python files because I got tired of having ten old experiments open at once and not knowing which one actually mattered.

## What is inside

- `hybrid_level3.py` — my local modified Level-3 pipeline. It detects the specific commuting-ZZ QAOA case I tested, uses V4 beam routing only in its current safe density window, then gives the routed circuit back to normal Qiskit Level 3. Everything else goes straight to stock Qiskit Level 3.
- `benchmark.py` — one benchmark runner. It checks correctness, runs both test suites, compares stock Level 3 vs Hybrid Level 3, optionally runs local hardware-calibrated noise simulation, and writes the report.
- `run.ps1` — the only script I normally need to touch.
- `requirements.txt` — versions used by this project.

I did **not** patch files inside `site-packages/qiskit`. That would be annoying to reproduce and easy to break after an update. `hybrid_level3.py` is the local Level-3 variant I can benchmark now. If this survives the final tests, the real next step for upstreaming is turning the same logic into a Qiskit contribution instead of secretly modifying an installed package.

## Run it

Open this folder in Cursor and use PowerShell.

First do the fast check:

```powershell
.\run.ps1 quick
```

That command does everything in one go: syntax check, balanced suite, workload suite, statevector correctness checks, stock-vs-hybrid comparison, and a tiny noise check so I know the noise path is not broken.

If quick finishes cleanly, the main benchmark is:

```powershell
.\run.ps1 standard
```

`standard` is the test I actually expect to use most. It runs 12 fresh instances for each random family, three compiler seeds, both `p=1` and `p=2`, both suites, correctness checks, and a limited noisy subset. Based on the PC where quick took about 75 seconds, I am targeting roughly **40-90 minutes**. That is an estimate, not a timer guarantee.

There is also a much heavier stress test:

```powershell
.\run.ps1 stress
```

`stress` is the heavy final run: more graphs, five compiler seeds, 512 noisy shots and a much larger noisy subset. I only need it if I want to really hammer the project after standard already looks good. Quick, standard and stress use different default graph ranges so I do not accidentally keep testing on graphs I already inspected.

If PowerShell blocks local scripts, the equivalent direct commands are:

```powershell
& '..\..\work\venv\Scripts\python.exe' .\benchmark.py --profile quick
& '..\..\work\venv\Scripts\python.exe' .\benchmark.py --profile standard
& '..\..\work\venv\Scripts\python.exe' .\benchmark.py --profile stress
```

You only run **one** of those at a time. Each one already runs every suite it needs.

| profile | what I use it for | rough target on my current PC |
|---|---|---:|
| `quick` | catch broken code before wasting time | ~1-3 min |
| `standard` | main evidence run | ~40-90 min |
| `stress` | heavy final stress test | several hours |

## Why there are two suites in the same run

I did not want to “discover” that V4 likes medium/dense graphs and then quietly make 80% of the benchmark medium/dense.

The **balanced suite** gives equal conceptual coverage to very sparse, sparse, medium, dense, near-complete, ring, and all-to-all interaction structures. The report has a family-balanced headline so ring/all-to-all do not disappear just because random families have more generated instances.

The **workload suite** uses cubic and quartic regular graphs, a 4×4 grid, a community graph, and Erdos-Renyi graphs at 25%, 50%, and 75% density. This is meant to be more like a mixed QAOA/MaxCut workload than a pure density sweep.

I am **not** claiming those are the measured frequencies of jobs on real IBM hardware. I do not have a dataset that proves, for example, “50%-dense QAOA appears 37% of the time,” so I refuse to invent one just because it would make the final percentage prettier. If I later get a real workload distribution, the benchmark can be weighted from actual data.

## What Hybrid Level 3 currently does

V4 only turns on when all of these are true:

- 16 logical qubits on the 16-qubit FakeGuadalupeV2-style setup used here;
- the circuit is made of full-width real commuting `ZZ` Pauli-evolution layers with one-qubit gates between them;
- every QAOA cost layer uses the same interaction graph;
- ZZ density is between **0.45 and 0.80**.

Outside that window Hybrid uses normal Qiskit Level 3. If V4 itself hits its search time budget or fails to finish a route, Hybrid also falls back to Qiskit.

I intentionally removed the old “pretend all-to-all is V4 but secretly run greedy” behavior. All-to-all and near-complete cases now say what actually happened: **Qiskit fallback**. That makes the benchmark less confusing.

## What the benchmark measures

For stock and Hybrid it records:

- native two-qubit gate count;
- CX count;
- total circuit depth;
- two-qubit depth;
- estimated scheduled duration from the historical `FakeGuadalupeV2` target;
- classical compilation time;
- selector decision / fallback reason;
- final logical-to-physical positions;
- statevector correctness error;
- noisy MaxCut score on the selected noisy subset.

The important distinction is that **estimated quantum-circuit duration** and **classical compiler runtime** are different things. Hybrid can make a circuit shorter on the fake quantum target while taking longer on the CPU to find that route. The report prints both instead of hiding one.

## Realism / limitations

The standard and stress benchmarks use `AerSimulator.from_backend(FakeGuadalupeV2)` for a noisy subset and use the same historical backend target for gate durations and connectivity. That is much closer to hardware than a bare graph-only CX count.

It is still not a current physical QPU run. A local fake backend does not perfectly reproduce crosstalk, calibration drift, queue/runtime behavior, or every hardware effect. So a good result here supports a **hardware-calibrated compiler result**, not “I sped up a real IBM quantum computer by X%.”

## AI use — I am not hiding this

I am 16 and I used AI heavily while making this project. AI wrote and rewrote a lot of the Python, helped me debug the benchmarks, and helped me reason about routing ideas and bad experimental setups. I chose what I wanted to build, ran the experiments, rejected approaches that failed, and kept iterating, but I did **not** personally type every line from scratch and I do not want to pretend I did.

If this ever becomes good enough for a real Qiskit PR, I would rather have the code judged on correctness, reproducibility and results than fake a story about how it was written.

## Results

Every run creates one folder under `results/` with only:

- `REPORT.md` — read this first;
- `summary.csv` — paired aggregate results;
- `raw.csv` — every compiler-seed run;
- `metadata.json` — exact versions and benchmark settings.

A positive “saving” means Hybrid used fewer gates / less depth / less estimated hardware time than stock Qiskit Level 3. A positive noisy MaxCut delta means Hybrid got the higher noisy cut score for the same logical problem, angles, shots and simulator seed.
