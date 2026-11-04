# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains a (slow) Python simulator.

It simulates a quantum circuit (an experiment) that has been compiled
to run on the simulator. It is exponential in the number of qubits.

The simulator is run using

.. plot::
   :include-source:
   :nofigs:

   BasicSimulator().run(run_input)

Where the input is a :class:`.QuantumCircuit` object and the output is a
:class:`.BasicProviderJob` object,
which can later be queried for the Result object. The result will contain a 'memory' data
field, which is a result of measurements for each shot.

"""

from __future__ import annotations

import math
import uuid
import time
import logging
import warnings
from collections import Counter

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping, GlobalPhaseGate
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.result import Result
from qiskit.transpiler import Target


from qiskit.quantum_info import Clifford, StabilizerState
from qiskit.exceptions import QiskitError


from .basic_provider_job import BasicProviderJob
from .basic_provider_tools import single_gate_matrix
from .basic_provider_tools import (
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
    TWO_QUBIT_GATES_WITH_PARAMETERS,
    THREE_QUBIT_GATES,
)
from .basic_provider_tools import einsum_vecmul_index
from .exceptions import BasicProviderError

logger = logging.getLogger(__name__)


class BasicSimulator(BackendV2):
    """Python implementation of a basic (non-efficient) quantum simulator.

    The simulator supports up to 24 qubits for statevector simulation and up to
    2048 qubits for Clifford/Stabilizer simulation."""

    # Formerly calculated as `int(log2(local_hardware_info()["memory"]*(1024**3)/16))`.
    # After the removal of `local_hardware_info()`, Statevector simulation is limited to 24 qubits.
    # Clifford/Stabilizer simulation can use 2048 qubits.
    # which matches the ~268 MB of required memory.

    MAX_QUBITS_STATEVECTOR = 24  # For statevector
    MAX_QUBITS_CLIFFORD = 2048  # For Clifford/Stabilizer simulation

    def __init__(
        self,
        provider=None,
        target: Target | None = None,
        **fields,
    ) -> None:
        """
        Args:
            provider: An optional backwards reference to the provider object that the backend
                is from.
            target: An optional target to configure the simulator.
            fields: kwargs for the values to use to override the default
                options.

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options.
        """

        super().__init__(
            provider=provider,
            name="basic_simulator",
            description="A Python simulator for basic quantum experiments",
            backend_version="0.1",
            **fields,
        )

        self._target = target

        # Internal simulator variables
        self._classical_memory = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._local_rng = None
        self._sample_measure = False
        self._shots = self.options.get("shots")
        self._memory = self.options.get("memory")
        self._initial_statevector = self.options.get("initial_statevector")
        self._seed_simulator = self.options.get("seed_simulator")
        self._use_clifford_optimization = self.options.get("use_clifford_optimization")

    @property
    def max_circuits(self) -> None:
        return None

    @property
    def target(self) -> Target:
        if not self._target:
            self._target = self._build_basic_target()
        return self._target

    def _build_basic_target(self) -> Target:
        """Helper method that returns a minimal target with a basis gate set but
        no coupling map or instruction properties.

        Returns:
            The configured target.
        """
        # Set num_qubits to None to signal the transpiler not to
        # resize the circuit to fit a specific (potentially too large)
        # number of qubits. The number of qubits in the circuits given to the
        # `run` method will determine the size of the simulated statevector.
        target = Target(
            description="Basic Target",
            num_qubits=None,
        )
        basis_gates = [
            "ccx",
            "ccz",
            "ch",
            "cp",
            "crx",
            "cry",
            "crz",
            "cs",
            "csdg",
            "cswap",
            "csx",
            "cu",
            "cu1",
            "cu3",
            "cx",
            "cy",
            "cz",
            "dcx",
            "delay",
            "ecr",
            "global_phase",
            "h",
            "id",
            "iswap",
            "measure",
            "p",
            "r",
            "rccx",
            "reset",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzx",
            "rzz",
            "s",
            "sdg",
            "swap",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "u",
            "u1",
            "u2",
            "u3",
            "unitary",
            "x",
            "xx_minus_yy",
            "xx_plus_yy",
            "y",
            "z",
        ]
        inst_mapping = get_standard_gate_name_mapping()
        for name in basis_gates:
            if name in inst_mapping:
                instruction = inst_mapping[name]
                target.add_instruction(instruction, properties=None, name=name)
            elif name == "unitary":
                # This is a placeholder for a UnitaryGate instance,
                # to signal the transpiler not to decompose unitaries
                # in the circuit.
                target.add_instruction(UnitaryGate, name="unitary")
            else:
                raise BasicProviderError(
                    f"Gate is not a valid basis gate for this simulator: {name}"
                )
        return target

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            memory=True,
            initial_statevector=None,
            seed_simulator=None,
            use_clifford_optimization=False,
        )

    def _add_unitary(self, gate: np.ndarray, qubits: list[int]) -> None:
        """Apply an N-qubit unitary matrix.

        Args:
            gate (matrix_like): an N-qubit unitary matrix
            qubits (list): the list of N-qubits.
        """
        # Get the number of qubits
        num_qubits = len(qubits)
        # Compute einsum index string for 1-qubit matrix multiplication
        indexes = einsum_vecmul_index(qubits, self._number_of_qubits)
        # Convert to complex rank-2N tensor
        gate_tensor = np.reshape(np.array(gate, dtype=complex), num_qubits * [2, 2])
        # Apply matrix multiplication
        self._statevector = np.einsum(
            indexes, gate_tensor, self._statevector, dtype=complex, casting="no"
        )

    def _get_measure_outcome(self, qubit: int) -> tuple[str, int]:
        """Simulate the outcome of measurement of a qubit.

        Args:
            qubit: index indicating the qubit to measure

        Returns:
            pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
        # Axis for numpy.sum to compute probabilities
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
        # Compute einsum index string for 1-qubit matrix multiplication
        random_number = self._local_rng.random()
        if random_number < probabilities[0]:
            return "0", probabilities[0]
        # Else outcome was '1'
        return "1", probabilities[1]

    def _add_sample_measure(
        self, measure_params: list[tuple[int, int]], num_samples: int
    ) -> list[hex]:
        """Generate memory samples from current statevector.

        Args:
            measure_params: List of (qubit, cmembit) values for
                                   measure instructions to sample.
            num_samples: The number of memory samples to generate.

        Returns:
            A list of memory values in hex format.
        """
        # Get unique qubits that are actually measured and sort in
        # ascending order
        measured_qubits = sorted({qubit for qubit, _ in measure_params})
        num_measured = len(measured_qubits)
        # We use the axis kwarg for numpy.sum to compute probabilities
        # this sums over all non-measured qubits to return a vector
        # of measure probabilities for the measured qubits
        axis = list(range(self._number_of_qubits))
        for qubit in reversed(measured_qubits):
            # Remove from largest qubit to smallest so list position is correct
            # with respect to position from end of the list
            axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.reshape(
            np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis)), 2**num_measured
        )
        # Generate samples on measured qubits as ints with qubit
        # position in the bit-string for each int given by the qubit
        # position in the sorted measured_qubits list
        samples = self._local_rng.choice(range(2**num_measured), num_samples, p=probabilities)
        # Convert the ints to bitstrings
        memory = []
        for sample in samples:
            classical_memory = self._classical_memory
            for qubit, cmembit in measure_params:
                pos = measured_qubits.index(qubit)
                qubit_outcome = int((sample & (1 << pos)) >> pos)
                membit = 1 << cmembit
                classical_memory = (classical_memory & (~membit)) | (qubit_outcome << cmembit)
            value = bin(classical_memory)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    def _add_measure(self, qubit: int, cmembit: int) -> None:
        """Apply a measure instruction to a qubit.

        Args:
            qubit: index of the qubit measured.
            cmembit: index of the classical memory bit to store outcome in.
        """
        # get measure outcome
        outcome, probability = self._get_measure_outcome(qubit)
        # update classical state
        membit = 1 << cmembit
        self._classical_memory = (self._classical_memory & (~membit)) | (int(outcome) << cmembit)

        # update quantum state
        if outcome == "0":
            update_diag = [[1 / math.sqrt(probability), 0], [0, 0]]
        else:
            update_diag = [[0, 0], [0, 1 / math.sqrt(probability)]]
        # update classical state
        self._add_unitary(update_diag, [qubit])

    def _add_reset(self, qubit: int) -> None:
        """Apply a reset instruction to a qubit.

        Args:
            qubit: the qubit being reset

        This is done by simulating a measurement
        outcome and projecting onto the outcome state while
        renormalizing.
        """
        # get measure outcome
        outcome, probability = self._get_measure_outcome(qubit)
        # update quantum state
        if outcome == "0":
            update = [[1 / math.sqrt(probability), 0], [0, 0]]
            self._add_unitary(update, [qubit])
        else:
            update = [[0, 1 / math.sqrt(probability)], [0, 0]]
            self._add_unitary(update, [qubit])

    def _validate_initial_statevector(self) -> None:
        """Validate an initial statevector"""
        # If the initial statevector isn't set we don't need to validate
        if self._initial_statevector is None:
            return
        # Check statevector is correct length for number of qubits
        length = len(self._initial_statevector)
        required_dim = 2**self._number_of_qubits
        if length != required_dim:
            raise BasicProviderError(
                f"initial statevector is incorrect length: {length} != {required_dim}"
            )

    def _set_run_options(self, run_options: dict | None = None) -> None:
        """Set the backend run options for all circuits"""

        # Reset internal variables every time "run" is called using saved options
        self._shots = self.options.get("shots")
        self._memory = self.options.get("memory")
        self._initial_statevector = self.options.get("initial_statevector")
        self._seed_simulator = self.options.get("seed_simulator")
        self._use_clifford_optimization = self.options.get("use_clifford_optimization")
        # Apply custom run options
        if run_options.get("initial_statevector", None) is not None:
            self._initial_statevector = np.array(run_options["initial_statevector"], dtype=complex)
        if self._initial_statevector is not None:
            # Check the initial statevector is normalized
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise BasicProviderError(f"Initial statevector is not normalized: norm {norm} != 1")
        if "shots" in run_options:
            self._shots = run_options["shots"]
        if "seed_simulator" in run_options:
            self._seed_simulator = run_options["seed_simulator"]
        elif self._seed_simulator is None:
            # For compatibility on Windows force dtype to be int32
            # and set the maximum value to be (2 ** 31) - 1
            self._seed_simulator = np.random.randint(2147483647, dtype="int32")
        if "memory" in run_options:
            self._memory = run_options["memory"]

        if "use_clifford_optimization" in run_options:
            self._use_clifford_optimization = run_options["use_clifford_optimization"]
        # Set seed for local random number gen.
        self._local_rng = np.random.default_rng(seed=self._seed_simulator)

    def _initialize_statevector(self) -> None:
        """Set the initial statevector for simulation"""
        if self._initial_statevector is None:
            # Set to default state of all qubits in |0>
            self._statevector = np.zeros(2**self._number_of_qubits, dtype=complex)
            self._statevector[0] = 1
        else:
            self._statevector = self._initial_statevector.copy()
        # Reshape to rank-N tensor
        self._statevector = np.reshape(self._statevector, self._number_of_qubits * [2])

    def _validate_measure_sampling(self, circuit: QuantumCircuit) -> None:
        """Determine if measure sampling is allowed for an experiment"""
        measure_flag = False
        # If shots=1 we should disable measure sampling.
        # This is also required for statevector simulator to return the
        # correct final statevector without silently dropping final measurements.
        if self._shots > 1:
            for instruction in circuit.data:
                # If circuit contains reset operations we cannot sample
                if instruction.name == "reset":
                    self._sample_measure = False
                    return
                # If circuit contains a measure option then we can
                # sample only if all following operations are measures
                if measure_flag:
                    # If we find a non-measure instruction
                    # we cannot do measure sampling
                    if instruction.name not in ["measure", "barrier", "id", "u0"]:
                        self._sample_measure = False
                        return
                elif instruction.name == "measure":
                    measure_flag = True
        self._sample_measure = measure_flag

    def run(
        self, run_input: QuantumCircuit | list[QuantumCircuit], **run_options
    ) -> BasicProviderJob:
        """Run on the backend.

        Args:
            run_input (QuantumCircuit or list): the QuantumCircuit (or list
                of QuantumCircuit objects) to run
            run_options (kwargs): additional runtime backend options

        Returns:
            BasicProviderJob: derived from BaseJob

        Additional Information:
            * kwarg options specified in ``run_options`` will temporarily override
              any set options of the same name for the current run. These may include:

                * "initial_statevector": vector-like. The "initial_statevector"
                  option specifies a custom initial statevector to be used instead
                  of the all-zero state. The size of this vector must correspond to
                  the number of qubits in the ``run_input`` argument.

                * "seed_simulator": int. This is the internal seed for sample
                  generation.

                * "shots": int. Number of shots used in the simulation.

                * "memory": bool. If True, the result will contain the results
                  of every individual shot simulation.

                * "use_clifford_optimization": bool. If True, enables Clifford
                  circuit optimization using stabilizer formalism. Default: False.

            Example::

                backend.run(
                    circuit_2q,
                    initial_statevector = np.array([1, 0, 0, 1j]) / math.sqrt(2)
                )
        """
        out_options = {}
        for key, value in run_options.items():
            if not hasattr(self.options, key):
                warnings.warn(
                    f"Option {key} is not used by this backend", UserWarning, stacklevel=2
                )
            else:
                out_options[key] = value
        self._set_run_options(run_options=run_options)
        job_id = str(uuid.uuid4())
        job = BasicProviderJob(self, job_id, self._run_job(job_id, run_input))
        return job

    def _run_job(self, job_id: str, run_input) -> Result:
        """Run circuits in run_input.

        Args:
            job_id: unique id for the job.
            run_input: circuits to be run.

        Returns:
            Result object
        """
        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]

        self._validate(run_input)
        result_list = []
        start = time.time()
        for circuit in run_input:
            result_list.append(self._run_circuit(circuit))
        end = time.time()
        result = {
            "backend_name": self.name,
            "backend_version": self.backend_version,
            "job_id": job_id,
            "results": result_list,
            "status": "COMPLETED",
            "success": True,
            "time_taken": (end - start),
        }

        return Result.from_dict(result)

    def _is_clifford_circuit(self, circuit: QuantumCircuit):
        """Check if circuit is Clifford and return Clifford object or None."""
        try:
            circ_no_meas = circuit.remove_final_measurements(inplace=False)
            return Clifford(circ_no_meas)
        except QiskitError:
            return None

    def _run_clifford_circuit(self, circuit: QuantumCircuit, clifford_obj) -> dict:
        """Simulate a Clifford circuit using StabilizerState.

        This method provides efficient simulation for Clifford circuits
        by using the stabilizer formalism instead of full statevector.

        Args:
            circuit: Clifford circuit to simulate
            clifford_obj: Pre-computed Clifford object representing the circuit

        Returns:
            Result dictionary matching _run_circuit format
        """

        start = time.time()

        # Find measurement operations
        measure_ops = []
        for operation in circuit.data:
            if operation.name == "measure":
                qubit = circuit.find_bit(operation.qubits[0]).index
                clbit = circuit.find_bit(operation.clbits[0]).index
                measure_ops.append((qubit, clbit))

        # Sample measurements
        memory = []
        if measure_ops:
            # Create StabilizerState once
            stab_state = StabilizerState(clifford_obj, validate=False)

            # Set seed if provided
            if self._seed_simulator is not None:
                stab_state.seed(self._seed_simulator)

            # Sample ALL shots at once (much faster than per-shot loop!)
            samples = stab_state.sample_memory(self._shots)

            # Process each sample
            for sample in samples:
                # Map measured qubits to classical bits
                classical_memory = 0
                for qubit, clbit in measure_ops:
                    bit_val = int(sample[-(qubit + 1)])
                    membit = 1 << clbit
                    classical_memory = (classical_memory & (~membit)) | (bit_val << clbit)

                # Convert to hex format
                outcome = bin(classical_memory)[2:]
                memory.append(hex(int(outcome, 2)))

        # Build result data
        data = {"counts": dict(Counter(memory))}
        if self._memory:
            data["memory"] = memory

        end = time.time()

        # Build header
        header = {
            "name": circuit.name,
            "n_qubits": circuit.num_qubits,
            "qreg_sizes": [[qreg.name, qreg.size] for qreg in circuit.qregs],
            "creg_sizes": [[creg.name, creg.size] for creg in circuit.cregs],
            "qubit_labels": [[qreg.name, j] for qreg in circuit.qregs for j in range(qreg.size)],
            "clbit_labels": [[creg.name, j] for creg in circuit.cregs for j in range(creg.size)],
            "memory_slots": circuit.num_clbits,
            "global_phase": circuit.global_phase,
            "metadata": circuit.metadata if circuit.metadata is not None else {},
        }

        return {
            "name": circuit.name,
            "seed_simulator": self._seed_simulator,
            "shots": self._shots,
            "data": data,
            "status": "DONE",
            "success": True,
            "header": header,
            "time_taken": (end - start),
        }

    def _run_circuit(self, circuit) -> dict:
        """Simulate a single circuit run.

        Args:
            circuit: circuit to be run.

        Returns:
             A result dictionary which looks something like::
                {
                "name": name of this experiment
                "seed": random seed used for simulation
                "shots": number of shots used in the simulation
                "header": {
                    "name": "circuit-206",
                    "n_qubits": 3,
                    "qreg_sizes": [['qr', 3]],
                    "creg_sizes": [['cr', 3]],
                    "qubit_labels": [['qr', 0], ['qr', 1], ['qr', 2]],
                    "clbit_labels": [['cr', 0], ['cr', 1], ['cr', 2]],
                    "memory_slots": 3,
                    "global_phase": 0.0,
                    "metadata": {},
                    }
                "data":
                    {
                    "counts": {'0x9: 5, ...},
                    "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                    },
                "status": status string for the simulation
                "success": boolean
                "time_taken": simulation time of this single experiment
                }
        Raises:
            BasicProviderError: if an error occurred.
        """

        # Set these BEFORE the Clifford check
        self._number_of_qubits = circuit.num_qubits
        self._number_of_cmembits = circuit.num_clbits

        # Check if circuit is Clifford and use optimized simulation
        # Check initial_statevector first (cheaper)
        if self._initial_statevector is None and self._use_clifford_optimization:
            clifford_obj = self._is_clifford_circuit(circuit)
            if clifford_obj is not None:
                return self._run_clifford_circuit(circuit, clifford_obj)

        start = time.time()

        self._number_of_qubits = circuit.num_qubits
        self._number_of_cmembits = circuit.num_clbits
        self._statevector = 0
        self._classical_memory = 0

        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()

        # Check if measure sampling is supported for current circuit
        self._validate_measure_sampling(circuit)

        # List of final counts for all shots
        memory = []
        # Check if we can sample measurements, if so we only perform 1 shot
        # and sample all outcomes from the final state vector
        if self._sample_measure:
            shots = 1
            # Store (qubit, cmembit) pairs for all measure ops in circuit to
            # be sampled
            measure_sample_ops = []
        else:
            shots = self._shots

        for _ in range(shots):
            self._initialize_statevector()
            # apply global_phase
            self._statevector *= np.exp(1j * circuit.global_phase)
            # Initialize classical memory to all 0
            self._classical_memory = 0

            for operation in circuit.data:
                if operation.name == "unitary":
                    qubits = [circuit.find_bit(bit).index for bit in operation.qubits]
                    gate = operation.operation.params[0]
                    self._add_unitary(gate, qubits)
                elif operation.name in ("id", "u0", "delay"):
                    pass
                elif operation.name == "global_phase":
                    params = getattr(operation, "params", None)
                    gate = GlobalPhaseGate(*params).to_matrix()
                    self._add_unitary(gate, [])
                # Check if single qubit gate
                elif operation.name in SINGLE_QUBIT_GATES:
                    params = getattr(operation, "params", None)
                    qubit = next(circuit.find_bit(bit).index for bit in operation.qubits)
                    gate = single_gate_matrix(operation.name, params)
                    self._add_unitary(gate, [qubit])
                elif operation.name in TWO_QUBIT_GATES_WITH_PARAMETERS:
                    params = getattr(operation, "params", None)
                    qubits = [circuit.find_bit(bit).index for bit in operation.qubits]
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    gate = TWO_QUBIT_GATES_WITH_PARAMETERS[operation.name](*params).to_matrix()
                    self._add_unitary(gate, [qubit0, qubit1])
                elif operation.name in ("id", "u0"):
                    pass
                elif operation.name in TWO_QUBIT_GATES:
                    qubits = [circuit.find_bit(bit).index for bit in operation.qubits]
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    gate = TWO_QUBIT_GATES[operation.name]
                    self._add_unitary(gate, [qubit0, qubit1])
                elif operation.name in THREE_QUBIT_GATES:
                    qubits = [circuit.find_bit(bit).index for bit in operation.qubits]
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    qubit2 = qubits[2]
                    gate = THREE_QUBIT_GATES[operation.name]
                    self._add_unitary(gate, [qubit0, qubit1, qubit2])
                # Check if reset
                elif operation.name == "reset":
                    qubits = [circuit.find_bit(bit).index for bit in operation.qubits]
                    qubit = qubits[0]
                    self._add_reset(qubit)
                # Check if barrier
                elif operation.name == "barrier":
                    pass
                # Check if measure
                elif operation.name == "measure":
                    qubit = next(circuit.find_bit(bit).index for bit in operation.qubits)
                    cmembit = next(circuit.find_bit(bit).index for bit in operation.clbits)
                    if self._sample_measure:
                        # If sampling measurements record the qubit and cmembit
                        # for this measurement for later sampling
                        measure_sample_ops.append((qubit, cmembit))
                    else:
                        # If not sampling perform measurement as normal
                        self._add_measure(qubit, cmembit)
                else:
                    backend = self.name
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise BasicProviderError(err_msg.format(backend, operation.name))

            # Add final creg data to memory list
            if self._number_of_cmembits > 0:
                if self._sample_measure:
                    # If sampling we generate all shot samples from the final statevector
                    memory = self._add_sample_measure(measure_sample_ops, self._shots)
                else:
                    # Turn classical_memory (int) into bit string and pad zero for unused cmembits
                    outcome = bin(self._classical_memory)[2:]
                    memory.append(hex(int(outcome, 2)))

        # Add counts to result data
        data = {"counts": dict(Counter(memory))}
        # Optionally, add memory list to result data
        if self._memory:
            data["memory"] = memory
        end = time.time()

        # Define header to be used by Result class to interpret counts
        header = {
            "name": circuit.name,
            "n_qubits": circuit.num_qubits,
            "qreg_sizes": [[qreg.name, qreg.size] for qreg in circuit.qregs],
            "creg_sizes": [[creg.name, creg.size] for creg in circuit.cregs],
            "qubit_labels": [[qreg.name, j] for qreg in circuit.qregs for j in range(qreg.size)],
            "clbit_labels": [[creg.name, j] for creg in circuit.cregs for j in range(creg.size)],
            "memory_slots": circuit.num_clbits,
            "global_phase": circuit.global_phase,
            "metadata": circuit.metadata if circuit.metadata is not None else {},
        }
        # Return result dictionary
        return {
            "name": circuit.name,
            "seed_simulator": self._seed_simulator,
            "shots": self._shots,
            "data": data,
            "status": "DONE",
            "success": True,
            "header": header,
            "time_taken": (end - start),
        }

    def _validate(self, run_input: list[QuantumCircuit]) -> None:
        """Semantic validations of the input."""
        for circuit in run_input:
            # Check which path: Clifford or Statevector
            use_clifford = (
                self._use_clifford_optimization and self._is_clifford_circuit(circuit) is not None
            )
            if use_clifford:
                max_qubits = self.MAX_QUBITS_CLIFFORD
            else:
                max_qubits = self.MAX_QUBITS_STATEVECTOR

            if circuit.num_qubits > max_qubits:
                raise BasicProviderError(
                    f"Number of qubits {circuit.num_qubits} is greater than maximum ({max_qubits}) "
                    f'for "{self.name}".'
                )
