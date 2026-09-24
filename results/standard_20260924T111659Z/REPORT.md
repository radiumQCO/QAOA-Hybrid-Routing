# Hybrid Level 3 benchmark

I wanted one test that makes it hard for me to accidentally cherry-pick a nice-looking result,
so this run always executes both suites in one command.

- **Balanced** deliberately covers very sparse through almost-complete graphs plus ring and all-to-all.
- **Workload** uses regular, grid, community and Erdos-Renyi interaction graphs. It is structure-diverse,
  but I am **not** pretending these are measured production frequencies from IBM hardware.

The comparison is stock Qiskit `optimization_level=3` vs my local Hybrid Level 3. Both get the exact
same logical PauliEvolutionGate QAOA circuit and compiler seed.

## Headline

### Balanced suite

- Family-balanced native 2Q saving: **4.45%**.
- Graph-weighted native 2Q saving: **6.03%** (95% bootstrap CI 4.55% to 7.55%).
- Estimated hardware duration saving: **6.91%**.
- 2Q-depth saving: **7.49%**.
- Classical compile-time change: **849.74%** (positive means Hybrid took longer to compile).
- Native-2Q W/T/L on graph-cases: **46/77/1**.
- Noisy MaxCut delta: **-0.212** across 4 paired noisy graph-cases.

### Workload suite

- Family-balanced native 2Q saving: **4.67%**.
- Graph-weighted native 2Q saving: **5.37%** (95% bootstrap CI 4.04% to 6.76%).
- Estimated hardware duration saving: **7.19%**.
- 2Q-depth saving: **7.67%**.
- Classical compile-time change: **862.78%** (positive means Hybrid took longer to compile).
- Native-2Q W/T/L on graph-cases: **48/98/0**.
- Noisy MaxCut delta: **0.406** across 4 paired noisy graph-cases.

## Per family / QAOA depth

| suite | family | p | graphs | native 2Q saving | duration saving | 2Q-depth saving | W/T/L | noisy cut delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | all_to_all | 1 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| balanced | all_to_all | 2 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| balanced | dense75 | 1 | 12 | 13.30% | 25.64% | 28.67% | 12/0/0 | n/a |
| balanced | dense75 | 2 | 12 | 13.88% | 15.47% | 18.46% | 12/0/0 | n/a |
| balanced | medium50 | 1 | 12 | 16.10% | 31.55% | 32.48% | 11/1/0 | -0.112 |
| balanced | medium50 | 2 | 12 | 19.06% | 24.09% | 25.20% | 11/0/1 | -0.737 |
| balanced | nearcomplete90 | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| balanced | nearcomplete90 | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| balanced | ring | 1 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| balanced | ring | 2 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| balanced | sparse25 | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | 0.000 |
| balanced | sparse25 | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | 0.000 |
| balanced | very_sparse15 | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| balanced | very_sparse15 | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | community | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | community | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | cubic | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | 0.000 |
| workload | cubic | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | 0.000 |
| workload | er25 | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | er25 | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | er50 | 1 | 12 | 16.91% | 34.58% | 35.23% | 12/0/0 | n/a |
| workload | er50 | 2 | 12 | 21.84% | 27.68% | 27.87% | 12/0/0 | n/a |
| workload | er75 | 1 | 12 | 12.28% | 22.79% | 25.77% | 12/0/0 | 1.570 |
| workload | er75 | 2 | 12 | 14.35% | 15.55% | 18.48% | 12/0/0 | 0.052 |
| workload | grid4x4 | 1 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| workload | grid4x4 | 2 | 1 | 0.00% | 0.00% | 0.00% | 0/1/0 | n/a |
| workload | quartic | 1 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |
| workload | quartic | 2 | 12 | 0.00% | 0.00% | 0.00% | 0/12/0 | n/a |

## What the selector actually did

- `qiskit_fallback`: 522 compiler-seed runs
- `v4_beam`: 288 compiler-seed runs

## Correctness and realism notes

- Every compiled circuit had to pass a statevector check with amplitude error <= 0.0001.
- Estimated duration comes from the historical FakeGuadalupeV2 target.
- Noise rows use AerSimulator.from_backend(FakeGuadalupeV2), so gate/readout/relaxation-style errors from
  that snapshot are included where Aer supports them.
- This still does not include every real effect such as all crosstalk, calibration drift, queue behavior,
  or a current device calibration. It is a hardware-calibrated local test, not a physical-QPU result.
- Profile: `standard`. Graph start: 80000. Compiler seeds: [11, 29, 47].

## Files

`raw.csv` has every compiler-seed run. `summary.csv` has the paired aggregate table. `metadata.json`
contains the exact profile and package versions.
