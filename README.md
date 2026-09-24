# QAOA Hybrid Routing

Experimental QAOA-aware routing combined with Qiskit Level 3.

This is my second attempt at improving QAOA compilation.

My first idea was much simpler: reorder commuting ZZ interactions before giving the circuit to SABRE. After testing it more carefully, I found that the original benchmark was not representative enough and the method did not consistently beat normal Qiskit Level 3.

So I dropped that approach and started again.

QAOA Hybrid Routing uses a specialized router for some commuting-ZZ QAOA circuits. Instead of only changing gate order, it searches over SWAP and layout choices directly.

The routing is used only for circuit structures where it has tested well. Other circuits fall back to normal Qiskit Level 3.

## Current result

The current standard benchmark uses:

- 16 logical qubits
- FakeGuadalupeV2
- QAOA p=1 and p=2
- 3 Qiskit compiler seeds
- 270 paired graph/case comparisons
- the same logical circuit and compiler seed for Stock and Hybrid

Results:

| Benchmark | Native 2Q reduction | Duration reduction | 2Q-depth reduction |
|-----------|--------------------:|-------------------:|-------------------:|
| Balanced  |               6.03% |              6.91% |              7.49% |
| Workload  |               5.37% |              7.19% |              7.67% |

When the specialized V4 router was activated, it had:

**94 wins / 1 tie / 1 loss**

in native two-qubit gate count across the tested graph-cases.

Depending on the interaction structure, the improvement on those activated cases was usually around 12–22%.

## Important limitation

The current router is slow.

It is mostly written in Python and its search takes much longer than stock Qiskit Level 3.

This is one of the main things I want to improve next. I am interested in moving the hot part of the router to Rust with AI(cuz idk nothing about Rust actually) and making the search itself cheaper.

## Correctness

Every compiled circuit in the benchmark had to pass a statevector equivalence check.

The benchmark uses the historical FakeGuadalupeV2 target. This is a local hardware-aware benchmark, not a result from a physical IBM quantum computer.

## Running the benchmark

Install the dependencies:

```powershell
pip install -r requirements.txt
