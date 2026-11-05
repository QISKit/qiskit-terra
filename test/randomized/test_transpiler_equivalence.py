# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Randomized tests of transpiler circuit equivalence.

This test can be optionally configured (e.g. by CI) via the
following env vars:

QISKIT_RANDOMIZED_TEST_LAYOUT_METHODS

    A space-delimited list of layout method names from which the
    randomizer should pick the layout method. Defaults to all
    available built-in methods if unspecified.

QISKIT_RANDOMIZED_TEST_ROUTING_METHODS

    A space-delimited list of routing method names from which the
    randomizer should pick the routing method. Defaults to all
    available built-in methods if unspecified.

QISKIT_RANDOMIZED_TEST_SCHEDULING_METHODS

    A space-delimited list of scheduling method names from which the
    randomizer should pick the scheduling method. Defaults to all
    available built-in methods if unspecified.

QISKIT_RANDOMIZED_TEST_BACKEND_NEEDS_DURATIONS

    A boolean value (e.g. "true", "Y", etc.) which, when true, forces
    the randomizer to pick a backend which fully supports scheduling
    (i.e. has fully specified duration info). Defaults to False.

QISKIT_RANDOMIZED_TEST_ALLOW_BARRIERS

    A boolean value (e.g. "true", "Y", etc.) which, when false,
    prevents the randomizer from emitting barrier instructions.
    Defaults to True.
"""

import os
import warnings

from test.utils.base import dicts_almost_equal

from math import pi

from hypothesis import assume, settings, HealthCheck
from hypothesis.stateful import multiple, rule, precondition, invariant
from hypothesis.stateful import Bundle, RuleBasedStateMachine
import hypothesis.strategies as st

from qiskit import transpile, qasm2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Measure, Reset, Barrier
from qiskit.providers.fake_provider import GenericBackendV2


from qiskit.circuit.library.standard_gates import (
    HGate,
    IGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CYGate,
    CZGate,
    SwapGate,
    CCXGate,
    CSwapGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    UGate,
    RZZGate,
    CPhaseGate,
    CUGate,
)
from ..python.legacy_cmaps import ALMADEN_CMAP, KYOTO_CMAP

from qiskit_aer import AerSimulator

default_profile = "transpiler_equivalence"
settings.register_profile(
    default_profile,
    report_multiple_bugs=False,
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", default_profile))

BASE_INSTRUCTIONS = {
    # Key is (n_qubits, n_clbits, n_params).  All gates here should be directly known by Aer so they
    # can be simulated without an initial transpile (whether that's via `execute` or not).
    (1, 0, 0): [HGate, IGate, SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate, Reset],
    (2, 0, 0): [CXGate, CYGate, CZGate, SwapGate],
    (3, 0, 0): [CCXGate, CSwapGate],
    (1, 0, 1): [PhaseGate, RXGate, RYGate, RZGate],
    (1, 0, 3): [UGate],
    (2, 0, 1): [RZZGate, CPhaseGate],
    (2, 0, 4): [CUGate],
    (1, 1, 0): [Measure],
}
variadic_gates = [Barrier]


def _strtobool(s):
    return s.lower() in ("y", "yes", "t", "true", "on", "1")


if not _strtobool(os.getenv("QISKIT_RANDOMIZED_TEST_ALLOW_BARRIERS", "True")):
    variadic_gates.remove(Barrier)


def _getenv_list(var_name):
    value = os.getenv(var_name)
    return None if value is None else value.split()


# Note: a value of `None` for any of the following methods means that
# the selected pass manager gets to choose. However, to avoid complexity,
# its not possible to specify `None` when overriding these with environment
# variables. Really, `None` is useful only for testing Terra's pass managers,
# and if you're overriding these, your goal is probably to test a specific
# pass or set of passes instead.
layout_methods = _getenv_list("QISKIT_RANDOMIZED_TEST_LAYOUT_METHODS") or [
    None,
    "trivial",
    "dense",
    "sabre",
]
routing_methods = _getenv_list("QISKIT_RANDOMIZED_TEST_ROUTING_METHODS") or [
    None,
    "basic",
    "lookahead",
    "sabre",
]
scheduling_methods = _getenv_list("QISKIT_RANDOMIZED_TEST_SCHEDULING_METHODS") or [
    None,
    "alap",
    "asap",
]

backend_needs_durations = _strtobool(
    os.getenv("QISKIT_RANDOMIZED_TEST_BACKEND_NEEDS_DURATIONS", "False")
)


mock_backend_configs = [
    {"num_qubits": 5, "seed": 0},
    {"num_qubits": 5, "seed": 2},
    {"num_qubits": 20, "coupling_map": ALMADEN_CMAP, "seed": 5},
    {"num_qubits": 127, "coupling_map": KYOTO_CMAP, "seed": 42},
]

mock_backend_configs_with_scheduling = mock_backend_configs


def _mock_backend(config):
    return GenericBackendV2(**config)


@st.composite
def transpiler_conf(draw):
    """Composite search strategy to pick a valid transpiler config."""
    all_backend_configs = st.one_of(st.none(), st.sampled_from(mock_backend_configs))
    scheduling_backend_configs = st.sampled_from(mock_backend_configs_with_scheduling)
    scheduling_method = draw(st.sampled_from(scheduling_methods))
    backend_config = (
        draw(scheduling_backend_configs)
        if scheduling_method or backend_needs_durations
        else draw(all_backend_configs)
    )
    return {
        "backend": None if backend_config is None else _mock_backend(backend_config),
        "optimization_level": draw(st.integers(min_value=0, max_value=3)),
        "layout_method": draw(st.sampled_from(layout_methods)),
        "routing_method": draw(st.sampled_from(routing_methods)),
        "scheduling_method": scheduling_method,
        "seed_transpiler": draw(st.integers(min_value=0, max_value=1_000_000)),
    }


class QCircuitMachine(RuleBasedStateMachine):
    """Build a Hypothesis rule based state machine for constructing, transpiling
    and simulating a series of random QuantumCircuits.

    Build circuits with up to QISKIT_RANDOM_QUBITS qubits, apply a random
    selection of gates from qiskit.circuit.library with randomly selected
    qargs, cargs, and parameters. At random intervals, transpile the circuit for
    a random backend with a random optimization level and simulate both the
    initial and the transpiled circuits to verify that their counts are the
    same.
    """

    qubits = Bundle("qubits")
    clbits = Bundle("clbits")

    backend = AerSimulator()
    max_qubits = int(backend.num_qubits / 2)

    # Limit reg generation for more interesting circuits
    max_qregs = 3
    max_cregs = 3

    def __init__(self):
        super().__init__()
        self.qc = QuantumCircuit()
        self.enable_variadic = bool(variadic_gates)

    @precondition(lambda self: len(self.qc.qubits) < self.max_qubits)
    @precondition(lambda self: len(self.qc.qregs) < self.max_qregs)
    @rule(target=qubits, n=st.integers(min_value=1, max_value=max_qubits))
    def add_qreg(self, n):
        """Adds a new variable sized qreg to the circuit, up to max_qubits."""
        n = min(n, self.max_qubits - len(self.qc.qubits))
        qreg = QuantumRegister(n)
        self.qc.add_register(qreg)
        return multiple(*list(qreg))

    @precondition(lambda self: len(self.qc.cregs) < self.max_cregs)
    @rule(target=clbits, n=st.integers(1, 5))
    def add_creg(self, n):
        """Add a new variable sized creg to the circuit."""
        creg = ClassicalRegister(n)
        self.qc.add_register(creg)
        return multiple(*list(creg))

    # Gates of various shapes

    @precondition(lambda self: self.qc.num_qubits > 0 and self.qc.num_clbits > 0)
    @rule(n_arguments=st.sampled_from(sorted(BASE_INSTRUCTIONS.keys())), data=st.data())
    def add_gate(self, n_arguments, data):
        """Append a random fixed gate to the circuit."""
        n_qubits, n_clbits, n_params = n_arguments
        gate_class = data.draw(st.sampled_from(BASE_INSTRUCTIONS[n_qubits, n_clbits, n_params]))
        qubits = data.draw(st.lists(self.qubits, min_size=n_qubits, max_size=n_qubits, unique=True))
        clbits = data.draw(st.lists(self.clbits, min_size=n_clbits, max_size=n_clbits, unique=True))
        params = data.draw(
            st.lists(
                st.floats(
                    allow_nan=False, allow_infinity=False, min_value=-10 * pi, max_value=10 * pi
                ),
                min_size=n_params,
                max_size=n_params,
            )
        )
        self.qc.append(gate_class(*params), qubits, clbits)

    @precondition(lambda self: self.enable_variadic)
    @rule(gate=st.sampled_from(variadic_gates), qargs=st.lists(qubits, min_size=1, unique=True))
    def add_variQ_gate(self, gate, qargs):
        """Append a gate with a variable number of qargs."""
        self.qc.append(gate(len(qargs)), qargs)

    # Properties to check
    @invariant()
    def qasm(self):
        """After each circuit operation, it should be possible to build QASM."""
        qasm2.dumps(self.qc)

    @precondition(lambda self: any(isinstance(d.operation, Measure) for d in self.qc.data))
    @rule(kwargs=transpiler_conf())
    def equivalent_transpile(self, kwargs):
        """Simulate, transpile and simulate the present circuit. Verify that the
        counts are not significantly different before and after transpilation.

        """
        assume(kwargs["backend"] is None or kwargs["backend"].num_qubits >= len(self.qc.qubits))

        call = (
            "transpile(qc, "
            + ", ".join(f"{key:s}={value!r}" for key, value in kwargs.items() if value is not None)
            + ")"
        )
        print(f"Evaluating {call} for:\n{qasm2.dumps(self.qc)}")  # noqa: T201

        shots = 4096

        # Note that there's no transpilation here, which is why the gates are limited to only ones
        # that Aer supports natively.
        with warnings.catch_warnings():
            # Safe to remove once https://github.com/Qiskit/qiskit-aer/pull/2179 is in a release version
            # of Aer.
            warnings.filterwarnings(
                "default",
                category=DeprecationWarning,
                module="qiskit_aer",
                message="Treating CircuitInstruction as an iterable",
            )
            aer_counts = self.backend.run(self.qc, shots=shots).result().get_counts()

        try:
            xpiled_qc = transpile(self.qc, **kwargs)
        except Exception as e:
            failed_qasm = (
                f"Exception caught during transpilation of circuit: \n{qasm2.dumps(self.qc)}"
            )
            raise RuntimeError(failed_qasm) from e

        xpiled_aer_counts = self.backend.run(xpiled_qc, shots=shots).result().get_counts()

        count_differences = dicts_almost_equal(aer_counts, xpiled_aer_counts, 0.05 * shots)

        assert count_differences == "", (
            f"Counts not equivalent: {count_differences}\nFailing QASM Input:\n"
            f"{qasm2.dumps(self.qc)}\n\nFailing QASM Output:\n{qasm2.dumps(xpiled_qc)}"
        )


TestQuantumCircuit = QCircuitMachine.TestCase
