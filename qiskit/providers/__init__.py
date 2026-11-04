# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=============================================
Providers Interface (:mod:`qiskit.providers`)
=============================================

.. currentmodule:: qiskit.providers

This module contains the classes used to build external providers for Qiskit. A
provider is anything that provides an external service to Qiskit. The typical
example of this is a Backend provider which provides
:class:`~qiskit.providers.Backend` objects which can be used for executing
:class:`~qiskit.circuit.QuantumCircuit`
objects. This module contains the abstract classes which are used to define the
interface between a provider and Qiskit.

Version Support
===============

Each providers interface abstract class is individually versioned. When we
need to make a change to an interface a new abstract class will be created to
define the new interface. These interface changes are not guaranteed to be
backwards compatible between versions.

Version Changes
----------------

Each minor version release of ``qiskit`` **may** increment the version of any
backend interface a single version number. It will be an aggregate of all
the interface changes for that release on that interface.

Version Support Policy
----------------------

To enable providers to have time to adjust to changes in this interface
Qiskit will support multiple versions of each class at once. Given the
nature of one version per release the version deprecation policy is a bit
more conservative than the standard deprecation policy. Qiskit will support a
provider interface version for a minimum of 3 minor releases or the first
release after 6 months from the release that introduced a version, whichever is
longer, prior to a potential deprecation. After that the standard deprecation
policy will apply to that interface version. This will give providers and users
sufficient time to adapt to potential breaking changes in the interface. So for
example lets say in 0.19.0 ``BackendV2`` is introduced and in the 3 months after
the release of 0.19.0 we release 0.20.0, 0.21.0, and 0.22.0, then 7 months after
0.19.0 we release 0.23.0. In 0.23.0 we can deprecate BackendV2, and it needs to
still be supported and can't be removed until the deprecation policy completes.

It's worth pointing out that Qiskit's version support policy doesn't mean
providers themselves will have the same support story, they can (and arguably
should) update to newer versions as soon as they can, the support window is
just for Qiskit's supported versions. Part of this lengthy window prior to
deprecation is to give providers enough time to do their own deprecation of a
potential end user impacting change in a user facing part of the interface
prior to bumping their version. For example, let's say we changed the signature
to ``Backend.run()`` in ``BackendV34`` in a backwards incompatible way. Before
Aer could update its :class:`~qiskit_aer.AerSimulator` class
to be based on version 34 they'd need to deprecate the old signature prior to switching
over. The changeover for Aer is not guaranteed to be lockstep with Qiskit, so we
need to ensure there is a sufficient amount of time for Aer to complete its
deprecation cycle prior to removing version 33 (ie making version 34
mandatory/the minimum version).

Abstract Classes
================

Backend
-------

.. autosummary::
   :toctree: ../stubs/

   Backend
   BackendV2
   QubitProperties

Options
-------

.. autosummary::
   :toctree: ../stubs/

   Options

Job
---

.. autosummary::
   :toctree: ../stubs/

   Job
   JobV1

Job Status
----------

.. autosummary::
   :toctree: ../stubs/

   JobStatus

Exceptions
----------

.. autoexception:: QiskitBackendNotFoundError
.. autoexception:: JobError
.. autoexception:: JobTimeoutError

Writing a New Backend
=====================

If you have a quantum device or simulator that you would like to integrate with
Qiskit you will need to write a backend. A provider is a collection of backends
and will provide Qiskit with a
method to get available :class:`~qiskit.providers.BackendV2` objects. The
:class:`~qiskit.providers.BackendV2` object provides both information describing
a backend and its operation for the :mod:`~qiskit.transpiler` so that circuits
can be compiled to something that is optimized and can execute on the
backend. It also provides the :meth:`~qiskit.providers.BackendV2.run` method which can
run the :class:`~qiskit.circuit.QuantumCircuit` objects. This enables users and other Qiskit
APIs to get results from
executing circuits on devices in a standard
fashion regardless of how the backend is implemented. At a high level the basic
steps for writing a provider are:

 * Implement a ``Provider`` class that handles access to the backend(s).
 * Implement a :class:`~qiskit.providers.BackendV2` subclass and its
   :meth:`~qiskit.providers.BackendV2.run` method.

   * Add any custom gates for the backend's basis to the session
     :class:`~qiskit.circuit.EquivalenceLibrary` instance.

 * Implement a :class:`~qiskit.providers.JobV1` subclass that handles
   interacting with a running job.

For a simple example of a provider, see the
`qiskit-aqt-provider <https://github.com/qiskit-community/qiskit-aqt-provider>`__

Provider
--------

A provider class serves a single purpose: to get backend objects that enable
executing circuits on a device or simulator. The expectation is that any
required credentials and/or authentication will be handled in the initialization
of a provider object. The provider object will then provide a list of backends,
and methods to filter and acquire backends (using the provided credentials if
required). An example provider class looks like::

    from qiskit.providers.providerutils import filter_backends

    from .backend import MyBackend

    class MyProvider:

        def __init__(self, token=None):
            super().__init__()
            self.token = token
            self.backends = [MyBackend(provider=self)]

        def backends(self, name=None, **kwargs):
            if name:
                backends = [
                    backend for backend in backends if backend.name() == name]
            return filter_backends(backends, filters=filters, **kwargs)

Ensure that any necessary information for
authentication (if required) is present in the class and that the backend's
method matches the required interface. The rest is up to the specific provider on how to implement.

Backend
-------

The backend classes are the core to the provider. These classes are what
provide the interface between Qiskit and the hardware or simulator that will
execute circuits. This includes providing the necessary information to
describe a backend to the compiler so that it can embed and optimize any
circuit for the backend. There are 4 required things in every backend object: a
:attr:`~qiskit.providers.BackendV2.target` property to define the model of the
backend for the compiler, a :attr:`~qiskit.providers.BackendV2.max_circuits`
property to define a limit on the number of circuits the backend can execute in
a single batch job (if there is no limit ``None`` can be used), a
:meth:`~qiskit.providers.BackendV2.run` method to accept job submissions, and
a :obj:`~qiskit.providers.BackendV2._default_options` method to define the
user configurable options and their default values. For example, a minimum working
example would be something like::

    from qiskit.providers import BackendV2 as Backend
    from qiskit.transpiler import Target
    from qiskit.providers import Options
    from qiskit.circuit import Parameter, Measure
    from qiskit.circuit.library import PhaseGate, SXGate, UGate, CXGate, IGate


    class Mybackend(Backend):

        def __init__(self):
            super().__init__()

            # Create Target
            self._target = Target("Target for My Backend")
            # Instead of None for this and below instructions you can define
            # a qiskit.transpiler.InstructionProperties object to define properties
            # for an instruction.
            lam = Parameter("λ")
            p_props = {(qubit,): None for qubit in range(5)}
            self._target.add_instruction(PhaseGate(lam), p_props)
            sx_props = {(qubit,): None for qubit in range(5)}
            self._target.add_instruction(SXGate(), sx_props)
            phi = Parameter("φ")
            theta = Parameter("ϴ")
            u_props = {(qubit,): None for qubit in range(5)}
            self._target.add_instruction(UGate(theta, phi, lam), u_props)
            cx_props = {edge: None for edge in [(0, 1), (1, 2), (2, 3), (3, 4)]}
            self._target.add_instruction(CXGate(), cx_props)
            meas_props = {(qubit,): None for qubit in range(5)}
            self._target.add_instruction(Measure(), meas_props)
            id_props = {(qubit,): None for qubit in range(5)}
            self._target.add_instruction(IGate(), id_props)

            # Set option validators
            self.options.set_validator("shots", (1, 4096))
            self.options.set_validator("memory", bool)

        @property
        def target(self):
            return self._target

        @property
        def max_circuits(self):
            return 1024

        @classmethod
        def _default_options(cls):
            return Options(shots=1024, memory=False)

        def run(self, circuits, **kwargs):
            # serialize circuits submit to backend and create a job
            for kwarg in kwargs:
                if not hasattr(self.options, kwarg):
                    warnings.warn(
                        "Option %s is not used by this backend" % kwarg,
                        UserWarning, stacklevel=2)
            options = {
                'shots': kwargs.get('shots', self.options.shots),
                'memory': kwargs.get('memory', self.options.memory),
            }
            job_json = convert_to_wire_format(circuit, options)
            job_handle = submit_to_backend(job_json)
            return MyJob(self.job_handle, job_json, circuit)


Backend's Transpiler Interface
------------------------------

The key piece of the :class:`~qiskit.providers.Backend` object is how it describes itself to the
compiler. This is handled with the :class:`~qiskit.transpiler.Target` class which defines
a model of a backend for the transpiler. A backend object will need to return
a :class:`~qiskit.transpiler.Target` object from the :attr:`~qiskit.providers.BackendV2.target`
attribute which the :func:`~qiskit.compiler.transpile` function will use as
its model of a backend target for compilation.

.. _custom_basis_gates:

Custom Basis Gates
^^^^^^^^^^^^^^^^^^

1. If your backend doesn't use gates in the Qiskit circuit library
   (:mod:`qiskit.circuit.library`) you can integrate support for this into your
   provider. The basic method for doing this is first to define a
   :class:`~qiskit.circuit.Gate` subclass for each custom gate in the basis
   set. For example::

       import numpy as np

       from qiskit.circuit import Gate
       from qiskit.circuit import QuantumCircuit

       class SYGate(Gate):
           def __init__(self, label=None):
               super().__init__("sy", 1, [], label=label)

           def _define(self):
               qc = QuantumCircuit(1)
               qc.ry(np.pi / 2, 0)
               self.definition = qc

   The key thing to ensure is that for any custom gates in your Backend's basis set
   your custom gate's name attribute (the first param on
   ``super().__init__()`` in the ``__init__`` definition above) does not conflict
   with the name of any other gates. The name attribute is what is used to
   identify the gate in the basis set for the transpiler. If there is a conflict
   the transpiler will not know which gate to use.

2. Add the custom gate to the target for your backend. This can be done with
   the :meth:`.Target.add_instruction` method. You'll need to add an instance of
   ``SYGate`` and its parameters to the target so the transpiler knows it
   exists. For example, assuming this is part of your
   :class:`~qiskit.providers.BackendV2` implementation for your backend::

       from qiskit.transpiler import InstructionProperties

       sy_props = {
           (0,): InstructionProperties(duration=2.3e-6, error=0.0002),
           (1,): InstructionProperties(duration=2.1e-6, error=0.0001),
           (2,): InstructionProperties(duration=2.5e-6, error=0.0003),
           (3,): InstructionProperties(duration=2.2e-6, error=0.0004),
       }
       self.target.add_instruction(SYGate(), sy_props)

   The keys in ``sy_props`` define the qubits where the backend ``SYGate`` can
   be used on, and the values define the properties of ``SYGate`` on that
   qubit. For multiqubit gates the tuple keys contain all qubit combinations
   the gate works on (order is significant, i.e. ``(0, 1)`` is different from
   ``(1, 0)``).

3. After you've defined the custom gates to use for the backend's basis set
   then you need to add equivalence rules to the standard equivalence library
   so that the :func:`~qiskit.compiler.transpile` function and
   :mod:`~qiskit.transpiler` module can convert an arbitrary circuit using the
   custom basis set. This can be done by defining equivalent circuits, in terms
   of the custom gate, for standard gates. Typically if you can convert from a
   :class:`~qiskit.circuit.library.CXGate` (if your basis doesn't include a
   standard 2 qubit gate) and some commonly used single qubit rotation gates like
   the :class:`~qiskit.circuit.library.HGate` and
   :class:`~qiskit.circuit.library.UGate` that should be sufficient for the
   transpiler to translate any circuit into the custom basis gates. But, the more
   equivalence rules that are defined from standard gates to your basis the more
   efficient translation from an arbitrary circuit to the target basis will be
   (although not always, and there is a diminishing margin of return).

   For example, if you were to add some rules for the above custom ``SYGate``
   we could define the :class:`~qiskit.circuit.library.U2Gate` and
   :class:`~qiskit.circuit.library.HGate`::

       from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
       from qiskit.circuit.library import HGate
       from qiskit.circuit.library import ZGate
       from qiskit.circuit.library import RZGate
       from qiskit.circuit.library import U2Gate


       # H => Z SY
       q = qiskit.QuantumRegister(1, "q")
       def_sy_h = qiskit.QuantumCircuit(q)
       def_sy_h.append(ZGate(), [q[0]], [])
       def_sy_h.append(SYGate(), [q[0]], [])
       SessionEquivalenceLibrary.add_equivalence(
           HGate(), def_sy_h)

       # u2 => Z SY Z
       phi = qiskit.circuit.Parameter('phi')
       lam = qiskit.circuit.Parameter('lambda')
       q = qiskit.QuantumRegister(1, "q")
       def_sy_u2 = qiskit.QuantumCircuit(q)
       def_sy_u2.append(RZGate(lam), [q[0]], [])
       def_sy_u2.append(SYGate(), [q[0]], [])
       def_sy_u2.append(RZGate(phi), [q[0]], [])
       SessionEquivalenceLibrary.add_equivalence(
           U2Gate(phi, lam), def_sy_u2)


   You will want this to be run on import so that as soon as the provider's
   package is imported it will be run. This will ensure that any time the
   :class:`~qiskit.transpiler.passes.BasisTranslator` pass is run with the
   custom gates the equivalence rules are defined.

   It's also worth noting that depending on the basis you're using, some optimization
   passes in the transpiler, such as
   :class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition`, may not
   be able to operate with your custom basis. For our ``SYGate`` example, the
   :class:`~qiskit.transpiler.passes.Optimize1qGatesDecomposition` will not be
   able to simplify runs of single qubit gates into the SY basis. This is because
   the :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` class does not
   know how to work in the SY basis. To solve this the ``SYGate`` class would
   need to be added to Qiskit and
   :class:`~qiskit.quantum_info.OneQubitEulerDecomposer` updated to support
   decomposing to the ``SYGate``. Longer term that is likely a better direction
   for custom basis gates and contributing the definitions and support in the
   transpiler will ensure that it continues to be well supported by Qiskit
   moving forward.

.. _custom_transpiler_backend:

Custom Transpiler Passes
^^^^^^^^^^^^^^^^^^^^^^^^
The transpiler supports the ability for backends to provide custom transpiler
stage implementations to facilitate hardware specific optimizations and
circuit transformations. Currently there are two stages supported,
``get_translation_stage_plugin()`` and ``get_scheduling_stage_plugin()``
which allow a backend to specify string plugin names to be used as the default
translation and scheduling stages, respectively. These
hook points in a :class:`~.BackendV2` class can be used if your
backend has requirements for compilation that are not met by the
current backend/:class:`~.Target` interface.  Please also consider
submitting a Github issue describing your use case as there is interest
in improving these interfaces to be able to describe more hardware
architectures in greater depth.

To leverage these hook points you just need to add the methods to your
:class:`~.BackendV2` implementation and have them return a string plugin name.
For example::


    class Mybackend(BackendV2):

        def get_scheduling_stage_plugin(self):
            return "SpecialDD"

        def get_translation_stage_plugin(self):
            return "BasisTranslatorWithCustom1qOptimization"

This snippet of a backend implementation will now have the :func:`~.transpile`
function use the ``SpecialDD`` plugin for the scheduling stage and
the ``BasisTranslatorWithCustom1qOptimization`` plugin for the translation
stage by default when the target is set to ``Mybackend``. Note that users may override these choices
by explicitly selecting a different plugin name. For this interface to work though transpiler
stage plugins must be implemented for the returned plugin name. You can refer
to :mod:`qiskit.transpiler.preset_passmanagers.plugin` module documentation for
details on how to implement plugins. The typical expectation is that if your backend
requires custom passes as part of a compilation stage the provider package will
include the transpiler stage plugins that use those passes. However, this is not
required and any valid method (from a built-in method or external plugin) can
be used.

This way if these two compilation steps are **required** for running or providing
efficient output on ``Mybackend`` the transpiler will be able to perform these
custom steps without any manual user input.

.. _providers-guide-real-time-variables:

Real-time variables
^^^^^^^^^^^^^^^^^^^

The transpiler will automatically handle real-time typed classical variables (see
:mod:`qiskit.circuit.classical`) and treat the :class:`.Store` instruction as a built-in
"directive", similar to :class:`.Barrier`.  No special handling from backends is necessary to permit
this.

If your backend is *unable* to handle classical variables and storage, we recommend that you comment
on this in your documentation, and insert a check into your :meth:`~.BackendV2.run` method (see
:ref:`providers-guide-backend-run`) to eagerly reject circuits containing them.  You can examine
:attr:`.QuantumCircuit.num_vars` for the presence of variables at the top level.  If you accept
:ref:`control-flow operations <circuit-control-flow-repr>`, you might need to recursively search the
internal :attr:`~.ControlFlowOp.blocks` of each for scope-local variables with
:attr:`.QuantumCircuit.num_declared_vars`.

For example, a function to check for the presence of any manual storage locations, or manual stores
to memory::

    from qiskit.circuit import Store, ControlFlowOp, QuantumCircuit

    def has_realtime_logic(circuit: QuantumCircuit) -> bool:
        if circuit.num_vars:
            return True
        for instruction in circuit.data:
            if isinstance(instruction.operation, Store):
                return True
            elif isinstance(instruction.operation, ControlFlowOp):
                for block in instruction.operation.blocks:
                    if has_realtime_logic(block):
                        return True
        return False

.. _angle-bounds-on-gates:

Angle bounds on Gates
^^^^^^^^^^^^^^^^^^^^^

If your backend has constraints on the allowed parameter values for any gate
in the target you can model this with angle bounds on the :class:`.Target`.
When you add the instruction with the :meth:`.add_instruction` you can use
the ``angle_bounds`` keyword argument which takes a list of tuples for the
upper and lower bound for the parameter of a gate.

For example, this code snippet instead of the example adding the :class:`.PhaseGate`
in the example above::

    lam = Parameter("λ")
    p_props = {(qubit,): None for qubit in range(5)}
    self._target.add_instruction(PhaseGate(lam), p_props, angle_bounds=[(0, math.pi)])

will set the bounds on :class:`.PhaseGate` to be between 0 and :math:`\\pi` (inclusive).
This models the angle constraint in the :class:`.Target` on the angle values for the
``lam`` parameter on :class:`.PhaseGate`. The :class:`.WrapAngles` transpiler pass is
used to transform any :class:`.PhaseGate` outside the specified angle bounds. You will
need to write a function that takes in the angle values for the gate and returns
a :class:`.DAGCircuit`. For example::

    from qiskit.transpiler.passes import WrapAngles

    def fold_phase(angles: List[float], qubits: List[int]) -> DAGCircuit:
        angle = angles[0]
        if angle > 0:
            number_of_gates = angle / math.pi
        else:
            number_of_gates = (6.28 - angle) / math.pi
        dag = DAGCircuit()
        dag.add_qubits([Qubit()])
        for _ in range(int(number_of_gates)):
            dag.apply_operation_back(PhaseGate(math.pi), [dag.qubits[0]])
        return dag

    WrapAngles.DEFAULT_REGISTRY.add_wrapper("phase", fold_phase)

This function will transform the out of bounds gates into one that respects the angle
bounds in the target and the target's other constraints (although not particularly well).

.. _providers-guide-backend-run:

Backend.run Method
------------------

Of key importance is the :meth:`~qiskit.providers.BackendV2.run` method, which
is used to actually submit circuits to a device or simulator. The run method
handles submitting the circuits to the backend to be executed and returning a
:class:`~qiskit.providers.Job` object. Depending on the type of backend this
typically involves serializing the circuit object into the API format used by a
backend.
Since backend serialization needs might differ (and, in the case of local
simulators, serialization may not even be needed), it is expected that the backend’s
:obj:`~qiskit.providers.BackendV2.run` method will handle this conversion.

An example run method would be something like::

    def run(self, circuits, **kwargs):
        for kwarg in kwargs:
            if not hasattr(self.options, kwarg):
                warnings.warn(
                    "Option %s is not used by this backend" % kwarg,
                    UserWarning, stacklevel=2)
        options = {
            'shots': kwargs.get('shots', self.options.shots),
            'memory': kwargs.get('memory', self.options.memory),
        }
        job_json = convert_to_wire_format(circuit, options)
        job_handle = submit_to_backend(job_json)
        return MyJob(self.job_handle, job_json, circuit)

Backend Options
---------------

There are often several options for a backend that control how a circuit is run.
The typical example of this is something like the number of ``shots`` which is
how many times the circuit is to be executed. The options available for a
backend are defined using an :class:`~qiskit.providers.Options` object. This
object is initially created by the
:obj:`~qiskit.providers.BackendV2._default_options` method of a Backend class.
The default options returns an initialized :class:`~qiskit.providers.Options`
object with all the default values for all the options a backend supports. For
example, if the backend only supports ``shots`` the
:obj:`~qiskit.providers.BackendV2._default_options` method would look like::

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

You can also set validators on an :class:`~qiskit.providers.Options` object to
provide limits and validation on user provided values based on what's acceptable
for your backend. For example, if the ``"shots"`` option defined above can be set
to any value between 1 and 4096 you can set the validator on the options object
for you backend with::

    self.options.set_validator("shots", (1, 4096))

you can refer to the :meth:`~qiskit.providers.Options.set_validator` documentation
for a full list of validation options.


Job
---

The output from the :obj:`~qiskit.providers.BackendV2.run` method is a :class:`~qiskit.providers.JobV1`
object. Each provider is expected to implement a custom job subclass that
defines the behavior for the provider. There are 2 types of jobs depending on
the backend's execution method, either a sync or async. By default jobs are
considered async and the expectation is that it represents a handle to the
async execution of the circuits submitted with ``Backend.run()``. An async
job object provides users the ability to query the status of the execution,
cancel a running job, and block until the execution is finished. The
:obj:`~qiskit.providers.JobV1.result` is the primary user facing method
which will block until the execution is complete and then will return a
:class:`~qiskit.result.Result` object with results of the job.

For some backends (mainly local simulators) the execution of circuits is a
synchronous operation and there is no need to return a handle to a running job
elsewhere. For sync jobs its expected that the
:obj:`~qiskit.providers.BackendV2.run` method on the backend will block until a
:class:`~qiskit.result.Result` object is generated and the sync job will return
with that inner :class:`~qiskit.result.Result` object.

An example job class for an async API based backend would look something like::

    from qiskit.providers import JobV1 as Job
    from qiskit.providers import JobError
    from qiskit.providers import JobTimeoutError
    from qiskit.providers.jobstatus import JobStatus
    from qiskit.result import Result


    class MyJob(Job):
        def __init__(self, backend, job_id, job_json, circuits):
            super().__init__(backend, job_id)
            self._backend = backend
            self.job_json = job_json
            self.circuits = circuits

        def _wait_for_result(self, timeout=None, wait=5):
            start_time = time.time()
            result = None
            while True:
                elapsed = time.time() - start_time
                if timeout and elapsed >= timeout:
                    raise JobTimeoutError('Timed out waiting for result')
                result = get_job_status(self._job_id)
                if result['status'] == 'complete':
                    break
                if result['status'] == 'error':
                    raise JobError('Job error')
                time.sleep(wait)
            return result

        def result(self, timeout=None, wait=5):
            result = self._wait_for_result(timeout, wait)
            results = [{'success': True, 'shots': len(result['counts']),
                        'data': result['counts']}]
            return Result.from_dict({
                'results': results,
                'backend_name': self._backend.configuration().backend_name,
                'backend_version': self._backend.configuration().backend_version,
                'job_id': self._job_id,
                'success': True,
            })

        def status(self):
            result = get_job_status(self._job_id)
            if result['status'] == 'running':
                status = JobStatus.RUNNING
            elif result['status'] == 'complete':
                status = JobStatus.DONE
            else:
                status = JobStatus.ERROR
            return status

        def submit(self):
            raise NotImplementedError

and for a sync job::

    class MySyncJob(Job):
        _async = False

        def __init__(self, backend, job_id, result):
            super().__init__(backend, job_id)
            self._result = result

        def submit(self):
            return

        def result(self):
            return self._result

        def status(self):
            return JobStatus.DONE

Primitives
----------

While not directly part of the provider interface, the :mod:`qiskit.primitives`
module is tightly coupled with providers. Specifically the primitive
interfaces, such as :class:`~.BaseSampler` and :class:`~.BaseEstimator`,
are designed to enable provider implementations to provide custom
implementations which are optimized for the provider's backends. This can
include customizations like circuit transformations, additional pre- and
post-processing, batching, caching, error mitigation, etc. The concept of
the :mod:`qiskit.primitives` module is to explicitly enable this as the
primitive objects are higher level abstractions to produce processed higher
level outputs (such as probability distributions and expectation values)
that abstract away the mechanics of getting the best result efficiently, to
concentrate on higher level applications using these outputs.

For example, if your backends were well suited to leverage
`mthree <https://github.com/Qiskit/qiskit-addon-mthree>`__ measurement
mitigation to improve the quality of the results, you could implement a
provider-specific :class:`~.Sampler` implementation that leverages the
``M3Mitigation`` class internally to run the circuits and return
quasi-probabilities directly from mthree in the result. Doing this would
enable algorithms to get the best results with
mitigation applied directly from your backends. You can refer to the
documentation in :mod:`qiskit.primitives` on how to write custom
implementations. Also, the built-in implementations: :class:`~.Sampler`,
:class:`~.Estimator`, :class:`~.BackendSampler`, and :class:`~.BackendEstimator`
can serve as references/models on how to implement these as well.
"""

# Providers interface
from qiskit.providers.backend import Backend
from qiskit.providers.backend import BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.options import Options
from qiskit.providers.job import Job
from qiskit.providers.job import JobV1

from qiskit.providers.exceptions import (
    JobError,
    JobTimeoutError,
    QiskitBackendNotFoundError,
)
from qiskit.providers.jobstatus import JobStatus

__all__ = [
    "Backend",
    "BackendV2",
    "Job",
    "JobError",
    "JobStatus",
    "JobTimeoutError",
    "JobV1",
    "Options",
    "QiskitBackendNotFoundError",
    "QubitProperties",
]
