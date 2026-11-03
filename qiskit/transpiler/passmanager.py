# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Manager for a set of Passes and their scheduling during transpilation."""
from __future__ import annotations

import inspect
import io
import re
from collections.abc import Iterator, Iterable, Callable
from functools import wraps
from typing import Any, TypeVar

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import BasePassManager, Task, FlowControllerLinear, PassManagerError
from .basepasses import BasePass
from .exceptions import TranspilerError
from .layout import TranspileLayout

_CircuitsT = TypeVar("_CircuitsT", bound=list[QuantumCircuit] | QuantumCircuit)


class PassManager(BasePassManager):
    """Manager for a set of Passes and their scheduling during transpilation."""

    def __init__(
        self,
        passes: Task | list[Task] = (),
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        super().__init__(
            tasks=passes,
            max_iteration=max_iteration,
        )

    def __repr__(self) -> str:
        """Return a representation of the PassManager.

        Returns:
            A string representation showing the number of pass sets, total passes,
            max iterations, and property information.
        """
        if self.property_set is None or len(self.property_set) == 0:
            properties = "0 properties"
        else:
            prop_keys = list(self.property_set.keys())
            nprops = len(prop_keys)
            if nprops > 3:
                _plist = "', '".join(prop_keys[0:3]) + "', ..."
            else:
                _plist = "', '".join(prop_keys) + "'"
            properties = f"{nprops} properties ('{_plist})"

        # _tasks is a list of lists of Task objects
        npass = sum(len(task_list) for task_list in self._tasks)

        return (
            f"<{type(self).__name__} with "
            f"{len(self._tasks)} sets, {npass} passes, "
            f"{self.max_iteration} max iterations, and {properties}>"
        )

    def _passmanager_frontend(
        self,
        input_program: QuantumCircuit,
        **kwargs,
    ) -> DAGCircuit:
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = {
                bit: i for i, bit in enumerate(input_program.qubits)
            }
        if self.property_set["num_input_qubits"] is None:
            self.property_set["num_input_qubits"] = input_program.num_qubits
        return circuit_to_dag(input_program, copy_operations=True)

    def _passmanager_backend(
        self,
        passmanager_ir: DAGCircuit,
        in_program: QuantumCircuit,
        **kwargs,
    ) -> QuantumCircuit:
        out_program = dag_to_circuit(passmanager_ir, copy_operations=False)
        if (out_name := kwargs.get("output_name", None)) is not None:
            out_program.name = out_name

        if (
            layout := TranspileLayout.from_property_set(passmanager_ir, self.property_set)
        ) is not None:
            out_program._layout = layout
            # Write the canonicalized form back out. This is for backwards compatibility.
            layout.write_into_property_set(self.property_set)

        out_program._clbit_write_latency = self.property_set["clbit_write_latency"]
        out_program._conditional_latency = self.property_set["conditional_latency"]
        if self.property_set["node_start_time"]:
            # This is dictionary keyed on the DAGOpNode, which is invalidated once
            # dag is converted into circuit. So this schedule information is
            # also converted into list with the same ordering with circuit.data.
            topological_start_times = []
            start_times = self.property_set["node_start_time"]
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            out_program._op_start_times = topological_start_times

        return out_program

    def append(  # pylint:disable=arguments-renamed
        self,
        passes: Task | list[Task],
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of transpiler passes to be added to schedule.

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
        """
        super().append(tasks=passes)

    def replace(  # pylint:disable=arguments-renamed
        self,
        index: int,
        passes: Task | list[Task],
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set to be added to the pass manager schedule.
        """
        super().replace(index, tasks=passes)

    def run(  # pylint:disable=arguments-renamed
        self,
        circuits: _CircuitsT,
        output_name: str | None = None,
        callback: Callable | None = None,
        num_processes: int | None = None,
        *,
        property_set: dict[str, object] | None = None,
    ) -> _CircuitsT:
        """Run all the passes on the specified ``circuits``.

        Args:
            circuits: Circuit(s) to transform via all the registered passes.
            output_name: The output circuit name. If ``None``, it will be set to the same as the
                input circuit name.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    dag (DAGCircuit): the dag output of the pass
                    time (float): the time to execute the pass
                    property_set (PropertySet): the property set
                    count (int): the index for the pass execution

                .. note::

                    Beware that the keyword arguments here are different to those used by the
                    generic :class:`.BasePassManager`.  This pass manager will translate those
                    arguments into the form described above.

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        pass_ = kwargs['pass_']
                        dag = kwargs['dag']
                        time = kwargs['time']
                        property_set = kwargs['property_set']
                        count = kwargs['count']
                        ...

                .. note::

                    When running transpilation with multi-processing,
                    the callback function is invoked within the context
                    of each sub-process, independently of the
                    parent process.

            num_processes: The maximum number of parallel processes to launch if parallel
                execution is enabled. This argument overrides ``num_processes`` in the user
                configuration file, and the ``QISKIT_NUM_PROCS`` environment variable. If set
                to ``None`` the system default or local user configuration will be used.
            property_set: If given, the initial value to use as the :class:`.PropertySet` for the
                pass manager pipeline.  This can be used to persist analysis from one run to
                another, in cases where you know the analysis is safe to share.  Beware that some
                analysis will be specific to the input circuit and the particular :class:`.Target`,
                so you should take a lot of care when using this argument.

        Returns:
            The transformed circuit(s).
        """
        if callback is not None:
            callback = _legacy_style_callback(callback)

        return super().run(
            in_programs=circuits,
            callback=callback,
            output_name=output_name,
            num_processes=num_processes,
            property_set=property_set,
        )

    def draw(self, filename=None, style=None, raw=False):
        """Draw the pass manager.

        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs
        `Graphviz <https://www.graphviz.org/>`__ to be installed.

        .. warning::
            This function will call the system Graphviz tool on a file involving user-controllable
            strings (such as pass names).  It is recommended to only call this function on trusted
            input.

        Args:
            filename (str): file path to save image to.
            style (dict): keys are the pass classes and the values are the colors to make them. An
                example can be seen in the DEFAULT_STYLE. An ordered dict can be used to ensure
                a priority coloring when pass falls into multiple categories. Any values not
                included in the provided dict will be filled in from the default dict.
            raw (bool): If ``True``, save the raw Dot output instead of the image.

        Returns:
            Optional[PassManager]: an in-memory representation of the pass manager, or ``None``
            if no image was generated or `Pillow <https://pypi.org/project/Pillow/>`__
            is not installed.

        Raises:
            ImportError: when nxpd or pydot not installed.
        """
        from qiskit.visualization import pass_manager_drawer

        return pass_manager_drawer(self, filename=filename, style=style, raw=raw)


class StagedPassManager(PassManager):
    """A pass manager pipeline built from individual stages.

    This class enables building a compilation pipeline out of fixed stages.
    Each ``StagedPassManager`` defines a list of stages which are executed in
    a fixed order, and each stage is defined as a standalone :class:`~.PassManager`
    instance. There are also ``pre_`` and ``post_`` stages for each defined stage.
    This enables easily composing and replacing different stages and also adding
    hook points to enable programmatic modifications to a pipeline. When using a staged
    pass manager you are not able to modify the individual passes and are only able
    to modify stages.

    By default, instances of ``StagedPassManager`` define a typical full compilation
    pipeline from an abstract virtual circuit to one that is optimized and
    capable of running on the specified backend. The default pre-defined stages are:

    #. ``init`` - Initial passes to run before embedding the circuit to the backend.
    #. ``layout`` - Maps the virtual qubits in the circuit to the physical qubits on
       the backend.
    #. ``routing`` - Inserts gates as needed to move the qubit states around until
       the circuit can be run with the chosen layout on the backend's coupling map.
    #. ``translation`` - Translates the gates in the circuit to the target backend's
       basis gate set.
    #. ``optimization`` - Optimizes the circuit to reduce the cost of executing it.
       These passes will typically run in a loop until a convergence criteria is met.
       For example, the convergence criteria might be that the circuit depth does not
       decrease in successive iterations.
    #. ``scheduling`` - Hardware-aware passes that schedule the operations in the
       circuit.

    .. note::

        For backwards compatibility the relative positioning of these default
        stages will remain stable moving forward. However, new stages may be
        added to the default stage list in between current stages. For example,
        in a future release a new phase, something like ``logical_optimization``, could be added
        immediately after the existing ``init`` stage in the default stage list.
        This would preserve compatibility for pre-existing ``StagedPassManager``
        users as the relative positions of the stage are preserved so the behavior
        will not change between releases.

    These stages will be executed in order and any stage set to ``None`` will be skipped.
    If a stage is provided multiple times (i.e. at different relative positions), the
    associated passes, including pre and post, will run once per declaration.
    If a :class:`~qiskit.transpiler.PassManager` input is being used for more than 1 stage here
    (for example in the case of a :class:`~.Pass` that covers both Layout and Routing) you will
    want to set that to the earliest stage in sequence that it covers.
    """

    invalid_stage_regex = re.compile(
        r"\s|\+|\-|\*|\/|\\|\%|\<|\>|\@|\!|\~|\^|\&|\:|\[|\]|\{|\}|\(|\)"
    )

    def __init__(self, stages: Iterable[str] | None = None, **kwargs) -> None:
        """Initialize a new StagedPassManager object

        Args:
            stages (Iterable[str]): An optional list of stages to use for this
                instance. If this is not specified the default stages list
                ``['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling']`` is
                used. After instantiation, the final list will be immutable and stored as tuple.
                If a stage is provided multiple times (i.e. at different relative positions), the
                associated passes, including pre and post, will run once per declaration.
            kwargs: The initial :class:`~.PassManager` values for any stages
                defined in ``stages``. If a argument is not defined the
                stages will default to ``None`` indicating an empty/undefined
                stage.

        Raises:
            AttributeError: If a stage in the input keyword arguments is not defined.
            ValueError: If an invalid stage name is specified.
        """
        stages = stages or [
            "init",
            "layout",
            "routing",
            "translation",
            "optimization",
            "scheduling",
        ]
        self._validate_stages(stages)
        # Set through parent class since `__setattr__` requires `expanded_stages` to be defined
        super().__setattr__("_stages", tuple(stages))
        super().__setattr__("_expanded_stages", tuple(self._generate_expanded_stages()))
        super().__init__()
        self._validate_init_kwargs(kwargs)
        for stage in set(self.expanded_stages):
            pm = kwargs.get(stage, None)
            setattr(self, stage, pm)

    def _validate_stages(self, stages: Iterable[str]) -> None:
        invalid_stages = [
            stage for stage in stages if self.invalid_stage_regex.search(stage) is not None
        ]
        if invalid_stages:
            with io.StringIO() as msg:
                msg.write(f"The following stage names are not valid: {invalid_stages[0]}")
                for invalid_stage in invalid_stages[1:]:
                    msg.write(f", {invalid_stage}")
                raise ValueError(msg.getvalue())

    def _validate_init_kwargs(self, kwargs: dict[str, Any]) -> None:
        expanded_stages = set(self.expanded_stages)
        for stage in kwargs.keys():
            if stage not in expanded_stages:
                raise AttributeError(f"{stage} is not a valid stage.")

    @property
    def stages(self) -> tuple[str, ...]:
        """Pass manager stages"""
        return self._stages

    @property
    def expanded_stages(self) -> tuple[str, ...]:
        """Expanded Pass manager stages including ``pre_`` and ``post_`` phases."""
        return self._expanded_stages

    def _generate_expanded_stages(self) -> Iterator[str]:
        for stage in self.stages:
            yield "pre_" + stage
            yield stage
            yield "post_" + stage

    def _update_passmanager(self) -> None:
        self._tasks = []
        for stage in self.expanded_stages:
            pm = getattr(self, stage, None)
            if pm is not None:
                self._tasks += pm._tasks

    def __setattr__(self, attr, value):
        if value == self and attr in self.expanded_stages:
            raise TranspilerError("Recursive definition of StagedPassManager disallowed.")
        super().__setattr__(attr, value)
        if attr in self.expanded_stages:
            self._update_passmanager()

    def append(
        self,
        passes: Task | list[Task],
    ) -> None:
        raise NotImplementedError

    def replace(
        self,
        index: int,
        passes: BasePass | list[BasePass],
    ) -> None:
        raise NotImplementedError

    # Raise NotImplementedError on individual pass manipulation
    def remove(self, index: int) -> None:
        raise NotImplementedError

    def __getitem__(self, index):
        self._update_passmanager()

        # Do not inherit from the PassManager, i.e. super()
        # It returns instance of self.__class__ which is StagedPassManager.
        new_passmanager = PassManager(max_iteration=self.max_iteration)
        new_passmanager._tasks = self._tasks[index]
        return new_passmanager

    def __len__(self):
        self._update_passmanager()
        return super().__len__()

    def __setitem__(self, index, item):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def run(
        self,
        circuits: _CircuitsT,
        output_name: str | None = None,
        callback: Callable | None = None,
        num_processes: int | None = None,
        *,
        property_set: dict[str, object] | None = None,
    ) -> _CircuitsT:
        self._update_passmanager()
        return super().run(circuits, output_name, callback, num_processes=num_processes)

    def to_flow_controller(self) -> FlowControllerLinear:
        self._update_passmanager()
        return super().to_flow_controller()

    def draw(self, filename=None, style=None, raw=False):
        """Draw the staged pass manager.

        .. warning::
            This function will call the system Graphviz tool on a file involving user-controllable
            strings (such as pass names).  It is recommended to only call this function on trusted
            input.

        Args:
            filename (str): file path to save image to.
            style (dict): keys are the pass classes and the values are the colors to make them. An
                example can be seen in the DEFAULT_STYLE. An ordered dict can be used to ensure
                a priority coloring when pass falls into multiple categories. Any values not
                included in the provided dict will be filled in from the default dict.
            raw (bool): If ``True``, save the raw Dot output instead of the image.
        """
        from qiskit.visualization import staged_pass_manager_drawer

        return staged_pass_manager_drawer(self, filename=filename, style=style, raw=raw)


# A temporary error handling with slight overhead at class loading.
# This method wraps all class methods to replace PassManagerError with TranspilerError.
# The pass flow controller mechanics raises PassManagerError, as it has been moved to base class.
# PassManagerError is not caught by TranspilerError due to the hierarchy.


def _replace_error(meth):
    @wraps(meth)
    def wrapper(*meth_args, **meth_kwargs):
        try:
            return meth(*meth_args, **meth_kwargs)
        except TranspilerError:
            # If it's already a `TranspilerError` subclass, don't erase the extra information.
            raise
        except PassManagerError as ex:
            raise TranspilerError(ex.message) from ex

    return wrapper


for _name, _method in inspect.getmembers(PassManager, predicate=inspect.isfunction):
    if _name.startswith("_"):
        # Ignore protected and private.
        # User usually doesn't directly execute and catch error from these methods.
        continue
    _wrapped = _replace_error(_method)
    setattr(PassManager, _name, _wrapped)


def _legacy_style_callback(callback: Callable):
    def _wrapped_callable(task, passmanager_ir, property_set, running_time, count):
        callback(
            pass_=task,
            dag=passmanager_ir,
            time=running_time,
            property_set=property_set,
            count=count,
        )

    return _wrapped_callable
