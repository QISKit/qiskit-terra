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

"""Base transpiler passes."""

from __future__ import annotations

import abc
from abc import abstractmethod
from collections.abc import Hashable, Iterable
from inspect import signature

from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, Callback
from qiskit.passmanager.compilation_status import PropertySet, RunState, PassManagerState

from .exceptions import TranspilerError


class MetaPass(abc.ABCMeta):
    """Metaclass for transpiler passes.

    Enforces the creation of some fields in the pass while allowing passes to
    override ``__init__``.
    """

    # Drop this functionality in the future.
    # This metaclass provides a pass equivalence evaluation based on the constructor arguments.
    # This implicit fake-hash based equivalence is fragile, and the pass developer must
    # explicitly implement equivalence check logic for each pass if necessary.
    # Currently, this metaclass is just here for backward compatibility, because
    # circuit pass manager has a functionality to avoid multiple execution of the
    # same pass (even though they are scheduled so). This is managed by the valid_passes set,
    # and executed passes are added to this collection to avoid future execution.
    # Dropping this metaclass causes many unittest failures and thus this is
    # considered as a breaking API change.
    # For example, test.python.transpiler.test_pass_scheduler.TestLogPasses.test_passes_in_linear

    def __call__(cls, *args, **kwargs):
        pass_instance = type.__call__(cls, *args, **kwargs)
        pass_instance._hash = hash(MetaPass._freeze_init_parameters(cls, args, kwargs))
        return pass_instance

    @staticmethod
    def _freeze_init_parameters(class_, args, kwargs):
        self_guard = object()
        init_signature = signature(class_.__init__)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = [("class_.__name__", class_.__name__)]
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)


class BasePass(GenericPass[DAGCircuit, DAGCircuit], metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        super().__init__()
        self.preserves: Iterable[GenericPass] = []
        self._hash = hash(None)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # Note that this implementation is incorrect.
        # This must be reimplemented in the future release.
        # See the discussion below for details.
        # https://github.com/Qiskit/qiskit/pull/10127#discussion_r1329982732
        return hash(self) == hash(other)

    @abstractmethod
    def run(self, dag: DAGCircuit):  # pylint:disable=arguments-renamed
        """Run a pass on the DAGCircuit. This is implemented by the pass developer.

        Args:
            dag: the dag on which the pass is run.

        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        """Check if the pass is a transformation pass.

        If the pass is a TransformationPass, that means that the pass can manipulate the DAG,
        but cannot modify the property set (but it can be read).
        """
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        """Check if the pass is an analysis pass.

        If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write
        the results of that analysis in the property set. Modifications on the DAG are not allowed
        by this kind of pass.
        """
        return isinstance(self, AnalysisPass)

    def __call__(
        self,
        circuit: QuantumCircuit,
        property_set: PropertySet | dict | None = None,
    ) -> QuantumCircuit:
        """Runs the pass on circuit.

        Args:
            circuit: The dag on which the pass is run.
            property_set: Input/output property set. An analysis pass might change the property set
                in-place.  If not given, the existing ``property_set`` attribute of the pass will
                be used (if set).

        Returns:
            If a transformation pass, the resulting QuantumCircuit.
            If analysis pass, the input circuit.
        """
        from qiskit.transpiler import PassManager

        pm = PassManager([self])
        # Previous versions of the `__call__` function would not construct a `PassManager`, but just
        # call `self.run` directly (this caused issues with `requires`).  It only overrode
        # `self.property_set` if the input was not `None`, which some users might have been relying
        # on (as our test suite was).
        if property_set is None:
            property_set = self.property_set
        out = pm.run(circuit, property_set=property_set)
        if property_set is not None and property_set is not pm.property_set:
            # When this `__call__` was first added, it contained this behaviour of mutating the
            # input `property_set` in-place, but didn't use the `PassManager` infrastructure.  This
            # preserves the output-variable nature of the `property_set` parameter.
            property_set.clear()
            property_set.update(pm.property_set)
        return out


class AnalysisPass(BasePass):
    """An analysis pass: change property set, not DAG."""


class TransformationPass(BasePass):
    """A transformation pass: change DAG, not property set."""

    def execute(
        self,
        passmanager_ir: DAGCircuit,
        state: DAGCircuit,
        callback: Callback[DAGCircuit] | None = None,
    ) -> tuple[DAGCircuit, PassManagerState]:
        new_dag, state = super().execute(
            passmanager_ir=passmanager_ir,
            state=state,
            callback=callback,
        )

        if state.workflow_status.previous_run == RunState.SUCCESS:
            if not isinstance(new_dag, DAGCircuit):
                raise TranspilerError(
                    "Transformation passes should return a transformed dag."
                    f"The pass {self.__class__.__name__} is returning a {type(new_dag)}"
                )

        return new_dag, state

    def update_status(
        self,
        state: PassManagerState,
        run_state: RunState,
    ) -> PassManagerState:
        state = super().update_status(state, run_state)
        if run_state == RunState.SUCCESS:
            state.workflow_status.completed_passes.intersection_update(set(self.preserves))
        return state
