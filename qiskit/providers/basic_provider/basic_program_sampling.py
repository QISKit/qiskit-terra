# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Answering one request of a quantum program by simulating the shot loops it holds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qiskit.quantum_program import InstructionRole
from qiskit.quantum_program.ops import ShotLoop

from .exceptions import BasicProviderError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_program import Request

    from .basic_simulator import BasicSimulator

#: The exclusive upper bound on a seed handed to the simulator, which takes a 32-bit one.
_SEED_LIMIT = 1 << 32


def answer_request(
    request: Request,
    simulator: BasicSimulator,
    generator: np.random.Generator,
) -> list[np.ndarray]:
    """Simulate every shot loop of ``request``'s function, in its declared result order.

    Args:
        request: The work to perform, whose function holds nothing but shot loops.
        simulator: The simulator to run each circuit on.
        generator: The source of each simulation's seed.

    Returns:
        One array per result the function declares, matching ``request.output_types``.

    Raises:
        BasicProviderError: If the function holds an operation other than a shot loop. Running
            a program cannot reach this, since partitioning on one op name puts nothing else in a
            function; answering a request built any other way can.
    """
    function = request.function
    # The instructions are in an order every one of them could run in, and a request's function
    # calls nothing, so every operand is already in hand by the time it is read.
    values: dict = dict(
        zip((instruction.outputs[0] for instruction in function.parameters), request.inputs)
    )
    for instruction in function:
        if instruction.role != InstructionRole.Op:
            continue
        op = instruction.op
        if not isinstance(op, ShotLoop):
            raise BasicProviderError(
                f"instruction {instruction.id} of the function to perform is "
                f"{instruction.full_name}, and this simulator performs only qiskit.shot_loop"
            )
        operands = [values[value] for value in instruction.operands]
        produced = sample_shot_loop(op, operands, simulator, generator)
        values.update(zip(instruction.outputs, produced))
    return [values[instruction.operands[0]] for instruction in function.results]


def sample_shot_loop(
    op: ShotLoop,
    operands: Sequence[np.ndarray],
    simulator: BasicSimulator,
    generator: np.random.Generator,
) -> list[np.ndarray]:
    """Run each circuit of ``op`` over the parameter values given for it.

    Args:
        op: The shot loop to perform.
        operands: One array of parameter values per circuit, in the order the shot loop takes them.
        simulator: The simulator to run each circuit on.
        generator: The source of each simulation's seed.

    Returns:
        One array per classical register of each circuit, in circuit-then-register order.
    """
    outcomes = []
    for circuit, values in zip(op.circuits(), operands):
        outcomes.extend(sample_circuit(circuit, op.shots, np.asarray(values), simulator, generator))
    return outcomes


def sample_circuit(
    circuit: QuantumCircuit,
    shots: int,
    values: np.ndarray,
    simulator: BasicSimulator,
    generator: np.random.Generator,
) -> list[np.ndarray]:
    """Run ``circuit`` once per parameter set in ``values``, collecting its registers separately.

    The leading axes of ``values`` are a batch of parameter sets, one simulation each, and its
    trailing axis holds one value per parameter of the circuit in the order
    :attr:`.QuantumCircuit.parameters` gives them. Those leading axes are carried onto each result.

    Args:
        circuit: The circuit to run.
        shots: How many shots to run it for.
        values: The parameter values, of shape ``(*batch, circuit.num_parameters)``.
        simulator: The simulator to run it on.
        generator: The source of each simulation's seed.

    Returns:
        One array of shape ``(*batch, shots, len(register))`` per classical register, in the order
        the circuit declares them. Element ``i`` of the trailing axis is bit ``i`` of that register.
    """
    cregs = circuit.cregs
    batch = values.shape[:-1]
    # A parameter-free circuit takes an empty trailing axis, and numpy cannot infer a row count
    # against one, so the rows are counted rather than inferred.
    rows = math.prod(batch)
    # A register's bits are neither contiguous nor in order in general, since a register may alias
    # bits another one holds, so each is gathered by position.
    positions = [
        np.array([circuit.find_bit(bit).index for bit in creg], dtype=np.intp) for creg in cregs
    ]
    # The dtype leaves each result a bit-valued tensor rather than a byte-valued one.
    blocks = [np.zeros((rows, shots, len(creg)), dtype=bool) for creg in cregs]

    # The simulator reports no memory at all for a circuit with no classical bit, so a circuit that
    # can record nothing is not run. Its results are empty either way.
    if circuit.num_clbits:
        for row, parameters in enumerate(values.reshape(rows, values.shape[-1])):
            bound = (
                circuit.assign_parameters(parameters.astype(float).tolist())
                if circuit.num_parameters
                else circuit
            )
            # Each row is seeded apart, so a batch of one parameter set repeated is still a batch of
            # independent samples.
            result = simulator.run(
                bound,
                shots=shots,
                memory=True,
                seed_simulator=int(generator.integers(_SEED_LIMIT)),
            ).result()
            outcomes = unpack_memory(result.data()["memory"], circuit.num_clbits)
            for block, columns in zip(blocks, positions):
                block[row] = outcomes[:, columns]

    return [block.reshape(batch + (shots, len(creg))) for block, creg in zip(blocks, cregs)]


def unpack_memory(memory: Sequence[str], num_clbits: int) -> np.ndarray:
    """Return the per-shot classical memory of a simulation, as one column per classical bit.

    Args:
        memory: One hexadecimal string per shot, as the simulator reports them.
        num_clbits: How many classical bits the circuit holds.

    Returns:
        An array of shape ``(len(memory), num_clbits)`` whose column ``i`` is classical bit ``i``.
    """
    width = -(-num_clbits // 8)
    buffer = b"".join(int(value, 16).to_bytes(width, "little") for value in memory)
    packed = np.frombuffer(buffer, dtype=np.uint8).reshape(len(memory), width)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :num_clbits].astype(bool)
