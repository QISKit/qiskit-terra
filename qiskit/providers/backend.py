# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Backend abstract interface for providers."""


from abc import ABC
from abc import abstractmethod
import datetime


from qiskit.circuit.gate import Instruction
from qiskit._accelerate.target import QubitProperties


class Backend:
    """Base common type for all versioned Backend abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """

    version = 0


class BackendV2(Backend, ABC):
    """Abstract class for Backends

    This abstract class is to be used for all Backend objects created by a
    provider. This version differs from earlier abstract Backend classes in
    that the configuration attribute no longer exists. Instead, attributes
    exposing equivalent required immutable properties of the backend device
    are added. For example ``backend.configuration().n_qubits`` is accessible
    from ``backend.num_qubits`` now.

    The ``options`` attribute of the backend is used to contain the dynamic
    user configurable options of the backend. It should be used more for
    runtime options that configure how the backend is used. For example,
    something like a ``shots`` field for a backend that runs experiments which
    would contain an int for how many shots to execute.

    A backend object can optionally contain methods named
    ``get_translation_stage_plugin`` and ``get_scheduling_stage_plugin``. If these
    methods are present on a backend object and this object is used for
    :func:`~.transpile` or :func:`~.generate_preset_pass_manager` the
    transpilation process will default to using the output from those methods
    as the scheduling stage and the translation compilation stage. This
    enables a backend which has custom requirements for compilation to specify a
    stage plugin for these stages to enable custom transformation of
    the circuit to ensure it is runnable on the backend. These hooks are enabled
    by default and should only be used to enable extra compilation steps
    if they are **required** to ensure a circuit is executable on the backend or
    have the expected level of performance. These methods are passed no input
    arguments and are expected to return a ``str`` representing the method name
    which should be a stage plugin (see: :mod:`qiskit.transpiler.preset_passmanagers.plugin`
    for more details on plugins). The typical expected use case is for a backend
    provider to implement a stage plugin for ``translation`` or ``scheduling``
    that contains the custom compilation passes and then for the hook methods on
    the backend object to return the plugin name so that :func:`~.transpile` will
    use it by default when targeting the backend.

    Subclasses of this should override the public method :meth:`run` and the internal
    :meth:`_default_options`:

    .. automethod:: _default_options
    """

    version = 2

    def __init__(
        self,
        provider=None,
        name: str | None = None,
        description: str | None = None,
        online_date: datetime.datetime | None = None,
        backend_version: str | None = None,
        **fields,
    ):
        """Initialize a BackendV2 based backend

        Args:
            provider: An optional backwards reference to the provider
                object that the backend is from
            name: An optional name for the backend
            description: An optional description of the backend
            online_date: An optional datetime the backend was brought online
            backend_version: An optional backend version string. This differs
                from the :attr:`~qiskit.providers.BackendV2.version` attribute
                as :attr:`~qiskit.providers.BackendV2.version` is for the
                abstract :attr:`~qiskit.providers.Backend` abstract interface
                version of the object while ``backend_version`` is for
                versioning the backend itself.
            fields: kwargs for the values to use to override the default
                options.

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options
        """

        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options:
                    raise AttributeError(f"Options field {field} is not valid for this backend")
            self._options.update_options(**fields)
        self.name = name
        """Name of the backend."""
        self.description = description
        """Optional human-readable description."""
        self.online_date = online_date
        """Date that the backend came online."""
        self.backend_version = backend_version
        """Version of the backend being provided.  This is not the same as
        :attr:`.BackendV2.version`, which is the version of the :class:`~.providers.Backend`
        abstract interface."""
        self._coupling_map = None

    @property
    def instructions(self) -> list[tuple[Instruction, tuple[int]]]:
        """A list of Instruction tuples on the backend of the form ``(instruction, (qubits))``"""
        return self.target.instructions

    @property
    def operations(self) -> list[Instruction]:
        """A list of :class:`~qiskit.circuit.Instruction` instances that the backend supports."""
        return list(self.target.operations)

    @property
    def operation_names(self) -> list[str]:
        """A list of instruction names that the backend supports."""
        return list(self.target.operation_names)

    @property
    @abstractmethod
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits the backend has."""
        return self.target.num_qubits

    @property
    def coupling_map(self):
        """Return the :class:`~qiskit.transpiler.CouplingMap` object"""
        if self._coupling_map is None:
            self._coupling_map = self.target.build_coupling_map()
        return self._coupling_map

    @property
    def instruction_durations(self):
        """Return the :class:`~qiskit.transpiler.InstructionDurations` object."""
        return self.target.durations()

    @property
    @abstractmethod
    def max_circuits(self):
        """The maximum number of circuits that can be
        run in a single job.

        If there is no limit this will return None
        """

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """

    @property
    def dt(self) -> float | None:
        """Return the system time resolution of input signals

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The input signal timestep in seconds. If the backend doesn't define ``dt``, ``None`` will
            be returned.
        """
        return self.target.dt

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            The output signal timestep in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                output signal timestep
        """
        raise NotImplementedError

    @property
    def meas_map(self) -> list[list[int]]:
        """Return the grouping of measurements which are multiplexed

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The grouping of measurements which are multiplexed

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    def qubit_properties(self, qubit: int | list[int]) -> QubitProperties | list[QubitProperties]:
        """Return QubitProperties for a given qubit.

        If there are no defined or the backend doesn't support querying these
        details this method does not need to be implemented.

        Args:
            qubit: The qubit to get the
                :class:`.QubitProperties` object for. This can
                be a single integer for 1 qubit or a list of qubits and a list
                of :class:`.QubitProperties` objects will be
                returned in the same order
        Returns:
            The :class:`~.QubitProperties` object for the
            specified qubit. If a list of qubits is provided a list will be
            returned. If properties are missing for a qubit this can be
            ``None``.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                qubit properties
        """
        # Since the target didn't always have a qubit properties attribute
        # to ensure the behavior here is backwards compatible with earlier
        # BackendV2 implementations where this would raise a NotImplemented
        # error.
        if self.target.qubit_properties is None:
            raise NotImplementedError
        if isinstance(qubit, int):
            return self.target.qubit_properties[qubit]
        return [self.target.qubit_properties[q] for q in qubit]

    def set_options(self, **fields):
        """Set the options fields for the backend

        This method is used to update the options of a backend. If
        you need to change any of the options prior to running just
        pass in the kwarg with the new value for the options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(f"Options field {field} is not valid for this backend")
        self._options.update_options(**fields)

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @property
    def provider(self):
        """Return the backend provider.

        Returns:
            provider: the provider responsible for the backend.
        """
        return self._provider

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method returns a :class:`~qiskit.providers.Job` object
        that runs circuits. Depending on the backend this may be either an async
        or sync call. It is at the discretion of the provider to decide whether
        running should block until the execution is finished or not: the Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or list): An
                individual or a list of :class:`.QuantumCircuit` objects to
                run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run
        """
