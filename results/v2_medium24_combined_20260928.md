# 24q medium route: combined check

Two fresh runs used graph IDs 99000–99004 and 99200–99209. The algorithm and its 10% depth selection rule were unchanged between runs. Together they contain 30 distinct 24-qubit QAOA circuits, each compiled on FakeBrisbane, FakeKyiv, and FakeSherbrooke: 90 paired compiler cases. These are local fake-backend compilations, not real-QPU measurements. The earlier 98000–98004 final holdout was not reused.

| Compiler | Routes built | Native 2Q gates | Total 2Q depth | Mean compile time |
| --- | ---: | ---: | ---: | ---: |
| Stock Level 3 | 90/90 | 116,676 | 35,508 | 0.166 s |
| Previous Hybrid Rust search | 31/90 | 112,591 | 32,210 | 0.987 s |
| New Hybrid line + beam tournament | **90/90** | **90,249** | **8,716** | **1.333 s** |
| Qiskit commuting + SAT | 90/90 | 98,265 | 8,970 | 3.052 s |

The new Hybrid used **8.2% fewer native 2Q gates** and **2.8% less summed 2Q depth** than the specialized Qiskit baseline. It won 69/90 paired cases by gate count; because each logical circuit was repeated on three targets, this represents 23/30 distinct logical circuits winning. All 90 selected Hybrid circuits came from the Rust optimized-line candidate. The previous Hybrid fell back to Stock in 59/90 cases, and its table totals include those fallbacks.

By layer count, the new Hybrid used 30,321 versus 31,521 2Q gates at `p=1` (3.8% fewer), and 59,928 versus 66,744 at `p=2` (10.2% fewer). Its `p=1` depth was 2,904 versus 2,880 (0.8% worse), while its `p=2` depth was 5,812 versus 6,090 (4.6% better). The advantage is therefore stronger at two layers and is not a depth win on every case.

Symbolic SWAP/gate-stream replay passed 90/90 optimized-line routes. Every recorded two-qubit instruction was supported by the corresponding target. No 24-qubit statevector or full compiled-output simulation was run. These checks support the routing logic but do not measure real-QPU fidelity.

Raw paired data: [first 30 cases](v2_medium24_rescue_20260928.csv) and [independent 60-case confirmation](v2_medium24_confirm_20260928.csv).
