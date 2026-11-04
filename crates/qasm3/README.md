# `qiskit._qasm3`

This crate is the Rust-level Qiskit interface to [a separately managed OpenQASM 3
parser](https://github.com/Qiskit/openqasm3_parser).  In order to maintain a sensible separation of
concerns, and because we hope to expand the use of that parser outside Qiskit, the parsing side does
not depend on Qiskit, and this crate interfaces with it in a Qiskit-specific manner to produce
`QuantumCircuit` objects.
