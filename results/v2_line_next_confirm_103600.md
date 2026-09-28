# v2.0.1 line router: fresh paired confirmation

Graph IDs 103600–103604 were not used for development. Frozen v2.0.0 compiles plain and optimized line candidates. The v2.0.1 path compiles one optimized line and, on medium graphs, ranks three 0.15-second Rust orders by symbolic SWAP count. Both use the same IBM fake target, native compiler, Qiskit seed, input circuit, and exactly the same seed-11 line order for each paired case. Timings include order search and native compilation, with verification outside the timer. The new routes pass symbolic SWAP/ZZ replay, final mapping, coupling, and native 2Q checks. No full 32/64q statevector or QPU run.

| Size / family | Cases | Next wins / ties / losses | Frozen 2Q | Next 2Q | 2Q change | Frozen depth | Next depth | Mean frozen / next time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64q dense75 | 30 | 0/30/0 | 259974 | 259974 | +0.0% | 8625 | 8625 | 3.04 / 1.46 s |
| 64q medium50 | 30 | 6/24/0 | 249825 | 248655 | -0.5% | 8267 | 8243 | 2.70 / 1.68 s |

Route and compiled mapping checks: 60/60 and 60/60. Peak process RAM: 565 MiB.

One medium50 case on FakeKyiv gained one 2Q-depth layer (183 to 184) while its native 2Q count fell from 5,631 to 5,545. The other 59 cases had equal or lower 2Q depth. These 60 target cases represent 20 distinct logical circuits compiled on three backend snapshots.
