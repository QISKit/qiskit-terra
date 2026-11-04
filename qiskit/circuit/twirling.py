# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The twirling module."""

from __future__ import annotations
import typing

from qiskit._accelerate.twirling import twirl_circuit as twirl_rs
from qiskit.circuit.quantumcircuit import QuantumCircuit, _copy_metadata
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import CXGate, ECRGate, CZGate, iSwapGate
from qiskit.exceptions import QiskitError

if typing.TYPE_CHECKING:
    from qiskit.transpiler.target import Target


NAME_TO_CLASS = {
    "cx": CXGate._standard_gate,
    "ecr": ECRGate._standard_gate,
    "cz": CZGate._standard_gate,
    "iswap": iSwapGate._standard_gate,
}


def pauli_twirl_2q_gates(
    circuit: QuantumCircuit,
    twirling_gate: None | str | Gate | list[str] | list[Gate] = None,
    seed: int | None = None,
    num_twirls: int | None = None,
    target: Target | None = None,
) -> QuantumCircuit | list[QuantumCircuit]:
    """Create copies of a given circuit with Pauli twirling applied around specified two qubit
    gates.

    If you're running this function with the intent to twirl a circuit to run on hardware this
    may not be the most efficient way to perform twirling. Especially if the hardware vendor
    has implemented the :mod:`.primitives` execution interface with :class:`.SamplerV2` and
    :class:`.EstimatorV2` this most likely is not the best way to apply twirling to your
    circuit and you'll want to refer to the implementation of :class:`.SamplerV2` and/or
    :class:`.EstimatorV2` for the specified hardware vendor.

    If the intent of this function is to be run after :func:`.transpile` or
    :meth:`.PassManager.run` the optional ``target`` argument can be used
    so that the inserted 1 qubit Pauli gates are synthesized to be
    compatible with the given :class:`.Target` so the output circuit(s) are
    still compatible.

    Args:
        circuit: The circuit to twirl
        twirling_gate: The gate to twirl, defaults to `None` which means twirl all default gates:
            :class:`.CXGate`, :class:`.CZGate`, :class:`.ECRGate`, and :class:`.iSwapGate`.
            If supplied it can either be a single gate or a list of gates either as either a gate
            object or its string name. Currently only the names `"cx"`, `"cz"`, `"ecr"`,  and
            `"iswap"` are supported. If a gate object is provided outside the default gates it must
            have a matrix defined from its :class:`~.Gate.to_matrix` method for the gate to potentially
            be twirled. If a valid twirling configuration can't be computed that particular gate will
            be silently ignored and not twirled.
        seed: An integer seed for the random number generator used internally by this function.
            If specified this must be between 0 and 18,446,744,073,709,551,615.
        num_twirls: The number of twirling circuits to build. This defaults to ``None`` and will return
            a single circuit. If it is an integer a list of circuits with `num_twirls` circuits
            will be returned.
        target: If specified an :class:`.Target` instance to use for running single qubit decomposition
            as part of the Pauli twirling to optimize and map the pauli gates added to the circuit
            to the specified target.

    Returns:
        A copy of the given circuit with Pauli twirling applied to each
        instance of the specified twirling gate.
    """
    custom_gates = None
    if isinstance(twirling_gate, str):
        gate = NAME_TO_CLASS.get(twirling_gate, None)
        if gate is None:
            raise QiskitError(f"The specified gate name {twirling_gate} is not supported")
        twirling_std_gate = [gate]
    elif isinstance(twirling_gate, list):
        custom_gates = []
        twirling_std_gate = []
        for gate in twirling_gate:
            if isinstance(gate, str):
                gate_name = gate
                gate = NAME_TO_CLASS.get(gate_name, None)
                if gate is None:
                    raise QiskitError(f"The specified gate name {gate_name} is not supported")
                twirling_std_gate.append(gate)
            else:
                twirling_gate = getattr(gate, "_standard_gate", None)

                if twirling_gate is None:
                    custom_gates.append(gate)
                elif twirling_gate in NAME_TO_CLASS.values():
                    twirling_std_gate.append(twirling_gate)
                else:
                    custom_gates.append(gate)
        if not custom_gates:
            custom_gates = None
        if not twirling_std_gate:
            twirling_std_gate = None
    elif twirling_gate is not None:
        std_gate = getattr(twirling_gate, "_standard_gate", None)
        if std_gate is None:
            twirling_std_gate = None
            custom_gates = [twirling_gate]
        elif std_gate in NAME_TO_CLASS.values():
            twirling_std_gate = [std_gate]
        else:
            twirling_std_gate = None
            custom_gates = [twirling_gate]
    else:
        twirling_std_gate = twirling_gate
    out_twirls = num_twirls
    if out_twirls is None:
        out_twirls = 1
    new_data = twirl_rs(
        circuit._data,
        twirling_std_gate,
        custom_gates,
        seed,
        out_twirls,
        target,
    )
    if num_twirls is not None:
        out_list = []
        for circ in new_data:
            new_circ = QuantumCircuit._from_circuit_data(circ)
            _copy_metadata(circuit, new_circ)
            out_list.append(new_circ)
        return out_list
    else:
        out_circ = QuantumCircuit._from_circuit_data(new_data[0])
        _copy_metadata(circuit, out_circ)
        return out_circ
