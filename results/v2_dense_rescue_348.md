# Dense completion on IBM fake targets

Paired MaxCut circuits on full targets: fake_brisbane, fake_kyiv, fake_sherbrooke. Stock, frozen Hybrid v2.0.0, and commuting + SAT counts come from the previous 348-case run. The new Hybrid uses a complete Rust line route, a 0.15 s Rust search for a better line ordering, and the existing 1 s Rust beam candidate. It fully compiles the available candidates and selects the fewest native 2Q gates. Medium-density cases still use the existing Hybrid path. This is a local compiler test, not a QPU result.

The same logical workloads are repeated across backends: 348 compiler cases cover 116 distinct logical circuits. Previous-run timings and this run's timings are indicative, not a simultaneous wall-clock race. A routed case means Rust produced a complete route without Stock fallback.

| Suite / size | Cases | Old routes | New routes | Stock 2Q | Old Hybrid 2Q | New Hybrid 2Q | Commuting 2Q | New vs commuting | New wins/ties/losses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full16 16q | 240 | 216 | 240 | 136353 | 113246 | 100840 | 107483 | -6.2% | 147/5/88 |
| scaling 16q | 36 | 34 | 36 | 20565 | 17281 | 15222 | 16394 | -7.1% | 22/2/12 |
| scaling 20q | 36 | 19 | 33 | 35412 | 32703 | 26717 | 25968 | +2.9% | 13/3/20 |
| scaling 24q | 36 | 10 | 28 | 53586 | 51345 | 41631 | 41718 | -0.2% | 23/0/13 |

## By graph family

| Suite / size | Family | Cases | Old routes | New routes | Old Hybrid 2Q | New Hybrid 2Q | Commuting 2Q | New vs commuting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full16 16q | dense75 | 60 | 46 | 60 | 34926 | 28204 | 28384 | -0.6% |
| full16 16q | er50 | 60 | 60 | 60 | 22477 | 22477 | 25192 | -10.8% |
| full16 16q | er75 | 60 | 50 | 60 | 34285 | 28601 | 28991 | -1.3% |
| full16 16q | medium50 | 60 | 60 | 60 | 21558 | 21558 | 24916 | -13.5% |
| scaling 16q | dense75 | 18 | 16 | 18 | 10471 | 8412 | 8805 | -4.5% |
| scaling 16q | medium50 | 18 | 18 | 18 | 6810 | 6810 | 7589 | -10.3% |
| scaling 20q | dense75 | 18 | 4 | 18 | 19753 | 13767 | 13731 | +0.3% |
| scaling 20q | medium50 | 18 | 15 | 15 | 12950 | 12950 | 12237 | +5.8% |
| scaling 24q | dense75 | 18 | 0 | 18 | 30150 | 20436 | 21333 | -4.2% |
| scaling 24q | medium50 | 18 | 10 | 10 | 21195 | 21195 | 20385 | +4.0% |

## Compile time and selection

| Suite / size | Median old Hybrid s | Median new Hybrid s | Median commuting s | Line / searched line / beam / existing / fallback |
|---|---:|---:|---:|---:|
| full16 16q | 0.150 | 0.285 | 0.824 | 0/113/7/120/0 |
| scaling 16q | 0.151 | 0.277 | 0.910 | 0/15/3/18/0 |
| scaling 20q | 0.600 | 1.125 | 2.671 | 0/18/0/15/3 |
| scaling 24q | 1.165 | 1.382 | 2.919 | 0/18/0/10/8 |

Peak process RSS: 254.2 MiB.
Per-case data: v2_dense_rescue_348.csv.
