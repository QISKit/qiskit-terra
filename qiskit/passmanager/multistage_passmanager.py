# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A staged passmanager supporting multiple IRs in the execution flow."""

from typing import Any, Generic
from collections.abc import Iterable

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, IR_OUT, PassManagerState, Callback
from .passmanager import BasePassManager
from .flow_controllers import FlowControllerLinear


class MultiStagePassManager(Generic[IR, IR_OUT]):
    """A staged pass manager supporting multiple IRs.

    This pass manager executes sequential, named stages on the input program.
    A stage can be defined as :class:`~.passmanager.Task`, an iterable thereof, or as
    a :class:`.BasePassManager`. If a :class:`.BasePassManager` is set as stage, only the tasks
    it contains are executed, the input and output conversions defined by its
    ``_passmanager_frontend`` and ``_passmanager_backend`` methods are _not_ applied.

    Stages can:

    * preserve the IR, for example if set as ``BasePassManager[IR]`` or ``Task[IR, IR]``, or
    * lower the IR, for example as ``Task[IR1, IR2]``.

    It is the user's responsibility to set up the stages in a compatible fashion, such that the
    output IR of the current stage matches in input IR of the next stage.  The stage names and the
    order they execute in is set when the pass manager is constructed.  The implementation of a
    stage can be modified by assigning a new implementation to the corresponding object attribute.

    If a callback is provided to the :meth:`run` method, it must be able to handle its IR input
    being any type output by the tasks in the pass manager.

    The callback is called with the signature::

        def callback(
            task: Task,  # the executed task
            passmanager_ir: Any,  # the IR after the task execution
            property_set: PropertySet,  # the property set after execution
            running_time: float,  # the time the task ran
            count: int  # the number of executed tasks so far
        ):
            ...

    All arguments are passed as keyword arguments.

    .. note::

        While :class:`~.passmanager.Task` object defines the task interface, custom passes should
        only derive from the base class :class:`.GenericPass`.  The ``Task`` base is an internal
        interface, and later Qiskit releases may place more restrictions on the available types of
        ``Task``.

    An example workflow is::

        from qiskit.circuit import QuantumCircuit
        from qiskit.dagcircuit import DAGCircuit
        from qiskit.passmanager import GenericPass, MultiStagePassManager
        from qiskit.transpiler import generate_preset_pass_manager, Target, CouplingMap

        class CustomPauliIR:
            # A custom IR of global Pauli strings
            def __init__(self, num_qubits):
                self.num_qubits = num_qubits
                self.paulis = []

            def apply(self, pauli: str):
                assert len(pauli) == self.num_qubits
                self.paulis.append(pauli)

        class CustomPauliOptimization(GenericPass[CustomPauliIR, CustomPauliIR]):
            # A pass run on the custom Pauli IR
            def run(self, passmanager_ir: CustomPauliIR) -> CustomPauliIR:
                to_remove = []
                for i, pauli in enumerate(passmanager_ir.paulis):
                    if all(p == "I" for p in pauli):
                        to_remove.append(i)

                for i in reversed(to_remove):
                    del passmanager_ir.paulis[i]

                return passmanager_ir

        class PauliToDAG(GenericPass[CustomPauliIR, DAGCircuit]):
            # A pass converting CustomPauliIR to DAGCircuit
            def run(self, passmanager_ir: CustomPauliIR) -> DAGCircuit:
                circuit = QuantumCircuit(passmanager_ir.num_qubits)
                for pauli in passmanager_ir.paulis:
                    circuit.pauli(pauli, circuit.qubits)
                return circuit.to_dag()

        def callback(task, passmanager_ir, property_set, running_time, count):
            if isinstance(passmanager_ir, CustomPauliIR):
                print("PauliIR:", task.__class__.__name__, passmanager_ir.paulis)
            else:
                print("DAGCircuit:", task.__class__.__name__, passmanager_ir.count_ops())

        target = Target.from_configuration(
            basis_gates=["u", "cx"], coupling_map=CouplingMap.from_line(3)
        )
        multi_pm = MultiStagePassManager(
            pauli_opt=CustomPauliOptimization(),
            pauli_to_dag=PauliToDAG(),
            dag_opt=generate_preset_pass_manager(target=target),
        )

        program = CustomPauliIR(3)
        program.apply("XYZ")
        program.apply("III")
        program.apply("ZZI")

        out = multi_pm.run(program, callback=callback)
        print(out.count_ops())


    This class relates to :class:`.StagedPassManager` in that both have a staged execution model.
    The :class:`.StagedPassManager`, however, only allows :class:`.DAGCircuit` as its IR,
    has implicit conversions from and to a :class:`.QuantumCircuit` at the input and output
    levels, and has implicit ``pre_*`` and ``post_*`` stage hooks that can be written into.

    The execution logic of a :class:`.StagedPassManager` is roughly equivalent to::

        from qiskit.circuit import QuantumCircuit
        from qiskit.dagcircuit import DAGCircuit
        from qiskit.passmanager import GenericPass, MultiStagePassManager
        from qiskit.transpiler import TranspileLayout

        class CircuitToDAG(GenericPass[QuantumCircuit, DAGCircuit]):
            def run(self, passmanager_ir: QuantumCircuit) -> DAGCircuit:
                self.property_set["original_qubit_indices"] = {
                    bit: i for i, bit in enumerate(passmanager_ir.qubits)
                }
                self.property_set["num_input_qubits"] = passmanager_ir.num_qubits
                return passmanager_ir.to_dag()

        class DAGToCircuit(GenericPass[DAGCircuit, QuantumCircuit]):
            def run(self, passmanager_ir: DAGCircuit) -> QuantumCircuit:
                qc = passmanager_ir.to_circuit(copy_operations=False)
                qc._layout = TranspileLayout.from_property_set(passmanager_ir, self.property_set)
                return qc

        multi_pm = MultiStagePassManager(
            input=CircuitToDAG(),
            # ... stages of StagedPassManager ...
            output=DAGToCircuit()
        )

        input_circuit = QuantumCircuit(1)
        output_circuit = multi_pm.run(input_circuit)

    .. warning::

        The current execution model linearizes the pass into a :class:`.FlowControllerLinear`
        to execute the tasks. This underlying model is subject to change and it is unsafe to
        build on this assumption. The public interfaces of this class, however, are stable.
    """

    def __init__(
        self,
        **stages: BasePassManager[Any] | Task[Any, Any] | Iterable[Task[Any, Any]],
    ):
        """
        Args:
            stages: The stages as pass managers. These will be executed in the provided order
                and must have compatible IRs.

                The stage names are fixed by the constructor; you cannot add new stages later, but
                you can replace the implementation of each stage by re-assigning to its attribute.
        """
        self._stages = tuple(stages)
        for name, stage in stages.items():
            setattr(self, name, stage)

    @property
    def stages(self) -> tuple[str, ...]:
        """The stage names. These are immutable.

        The stages themselves can be modified by writing to the attribute with the same name
        as the stage.
        """
        return self._stages

    def to_flow_controller(self) -> FlowControllerLinear[IR, IR_OUT]:
        """Convert this multi-staged pass manager to a linear flow controller.

        This conversion normalizes this pass manager into a ``Task[IR, IR_OUT]`` and allows
        it to be nested inside a :class:`.MultiStagePassManager` itself or other execution flows.
        """
        tasks = []
        for name in self.stages:
            stage = getattr(self, name)
            if isinstance(stage, BasePassManager):
                tasks.extend(stage.to_flow_controller().tasks)
            elif isinstance(stage, Task):
                tasks.append(stage)
            else:  # case: Iterable[Task]
                tasks.extend(stage)

        return FlowControllerLinear(tasks)

    def run(
        self,
        in_programs: IR | list[IR] | tuple[IR],
        callback: Callback[Any] | None = None,
        *,
        property_set: PropertySet | None = None,
    ) -> IR_OUT | Iterable[IR_OUT]:
        """Run the pass manager on a set of input programs.

        Args:
            in_programs: The programs to run the pass manager on.
            callback: A callback passed to each individual task.
            property_set: An optional property set to pass into the pass manager.  This will be
                mutated in place, if given.  This cannot be used with multiple in programs.

        Returns:
            The output programs.
        """
        controller = self.to_flow_controller()

        def state(property_set):
            return PassManagerState(property_set=property_set, workflow_status=WorkflowStatus())

        if isinstance(in_programs, (list, tuple)):
            if property_set is not None:
                raise ValueError(
                    "a 'property_set' cannot be provided when passing multiple input programs"
                )
            return [
                controller.execute(program, state(PropertySet()), callback)[0]
                for program in in_programs
            ]

        property_set = PropertySet() if property_set is None else property_set
        return controller.execute(in_programs, state(property_set), callback)[0]
