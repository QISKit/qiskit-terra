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

"""Tests for running a quantum program on a state vector."""

from unittest import mock

import numpy as np

from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit, QuantumRegister
from qiskit.providers.basic_provider import (
    BasicProgramJob,
    BasicProgramRuntime,
    BasicProgramSimulator,
    BasicProviderError,
    BasicSimulator,
)
from qiskit.providers.basic_provider.basic_program_runtime import replay
from qiskit.providers.basic_provider.basic_program_sampling import (
    answer_request,
    sample_circuit,
    unpack_memory,
)
from qiskit.providers.jobstatus import JobStatus
from qiskit.quantum_program import DataTree, ProgramStepper, build, f64, qp_input, shot_loop
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def flipping_circuit():
    """A circuit setting q0 and q2, with registers `a` over q0 and `b` over q1 and q2."""
    qr = QuantumRegister(3, "q")
    a, b = ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
    circuit = QuantumCircuit(qr, a, b)
    circuit.x(0)
    circuit.x(2)
    circuit.measure(0, a[0])
    circuit.measure([1, 2], [b[0], b[1]])
    return circuit


def swept_circuit():
    """A one-parameter circuit whose excitation is `sin(theta / 2) ** 2`."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)
    return circuit


def sample(circuit, shots=4, values=None, seed=7):
    """Run `circuit` through `sample_circuit`, defaulting to one parameter-free simulation."""
    if values is None:
        values = np.zeros(circuit.num_parameters)
    return sample_circuit(
        circuit,
        shots,
        np.asarray(values, dtype=float),
        BasicSimulator(),
        np.random.default_rng(seed),
    )


class TestUnpackMemory(QiskitTestCase):
    """Tests for reading a simulation's classical memory as one column per bit."""

    def test_bit_i_is_clbit_i(self):
        """Test that column i of the unpacked memory is classical bit i."""
        unpacked = unpack_memory(["0x5", "0x2"], 3)
        np.testing.assert_array_equal(unpacked, [[True, False, True], [False, True, False]])
        self.assertEqual(unpacked.dtype, np.dtype(bool))

    def test_memory_wider_than_64_bits(self):
        """Test that a circuit of more than 64 classical bits is unpacked whole."""
        unpacked = unpack_memory([hex(1 << 96)], 100)
        self.assertEqual(unpacked.shape, (1, 100))
        self.assertEqual(list(np.flatnonzero(unpacked[0])), [96])

    def test_no_shots_gives_no_rows(self):
        """Test that empty memory unpacks to no rows."""
        self.assertEqual(unpack_memory([], 3).shape, (0, 3))


class TestSampleCircuit(QiskitTestCase):
    """Tests for running one circuit over a batch of parameter sets."""

    def test_bits_are_in_register_order(self):
        """Test that element i of a result's trailing axis is bit i of that register."""
        a, b = sample(flipping_circuit())
        self.assertEqual((a.shape, b.shape), ((4, 1), (4, 2)))
        np.testing.assert_array_equal(a, np.full((4, 1), True))
        np.testing.assert_array_equal(b, np.tile([False, True], (4, 1)))

    def test_an_aliased_register_reads_its_bits(self):
        """Test that a register aliasing another's bits reports them in its own order."""
        circuit = flipping_circuit()
        reversed_b = ClassicalRegister(bits=[circuit.cregs[1][1], circuit.cregs[1][0]])
        circuit.add_register(reversed_b)
        _, b, alias = sample(circuit)
        np.testing.assert_array_equal(alias, b[:, ::-1])

    def test_parameters_bind_in_circuit_order(self):
        """Test that the trailing axis is in QuantumCircuit.parameters order."""
        # Written second but sorted first, so a positional bind that used insertion order would
        # flip the wrong qubit.
        first, second = Parameter("b"), Parameter("a")
        circuit = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
        circuit.ry(first, 0)
        circuit.ry(second, 1)
        circuit.measure([0, 1], [0, 1])
        self.assertEqual([p.name for p in circuit.parameters], ["a", "b"])

        [bits] = sample(circuit, shots=2, values=[0.0, np.pi])
        np.testing.assert_array_equal(bits, np.tile([True, False], (2, 1)))

    def test_batch_axes_are_kept(self):
        """Test that the leading axes of the parameter values are carried onto each result."""
        [bits] = sample(swept_circuit(), shots=8, values=np.zeros((4, 3, 1)))
        self.assertEqual(bits.shape, (4, 3, 8, 1))

    def test_batch_rows_are_seeded_apart(self):
        """Test that repeating one parameter set still samples each row independently."""
        # One seed for the whole batch would make every row of shots identical.
        [bits] = sample(swept_circuit(), shots=64, values=np.full((8, 1), np.pi / 2))
        self.assertGreater(len({row.tobytes() for row in bits}), 1)

    def test_a_parameter_free_circuit_runs(self):
        """Test that a circuit taking no parameters has an empty trailing axis and no batch."""
        [bits] = sample(flipping_circuit(), values=np.zeros(0))[1:]
        self.assertEqual(bits.shape, (4, 2))

    def test_a_zero_size_batch_gives_no_rows(self):
        """Test that a batch of no parameter sets runs nothing and returns no rows."""
        [bits] = sample(swept_circuit(), shots=8, values=np.zeros((0, 1)))
        self.assertEqual(bits.shape, (0, 8, 1))

    def test_a_circuit_with_no_register_gives_nothing(self):
        """Test that a circuit declaring no classical register produces no results."""
        circuit = QuantumCircuit(QuantumRegister(1, "q"))
        circuit.h(0)
        self.assertEqual(sample(circuit), [])

    def test_a_zero_width_register_gives_no_bits(self):
        """Test that a register of no bits produces a result of no bits."""
        circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(0, "c"))
        circuit.h(0)
        [bits] = sample(circuit, shots=8)
        self.assertEqual(bits.shape, (8, 0))


def sampling_program(circuits, shots=64, **build_outputs):
    """A program partitioned so that its shot loop is the caller's to perform.

    Returns the partitioned program and the positions of the functions to perform.
    """
    outcomes = shot_loop(circuits, shots=shots)
    program = build(build_outputs or {"bits": outcomes})
    program, resources = program.partition([["qiskit.shot_loop"]])
    return program, range(len(resources))


class TestAnswerRequest(QiskitTestCase):
    """Tests for answering one request of an stepper."""

    def answer(self, stepper, seed=7):
        """The answer to the one request `stepper` has raised, and that request."""
        request = stepper.outstanding()[0]
        generator = np.random.default_rng(seed)
        return request, answer_request(request, BasicSimulator(), generator)

    def test_a_multi_circuit_loop_answers_in_order(self):
        """Test that a shot loop over several circuits answers circuit then register."""
        narrow = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "n"))
        narrow.x(0)
        narrow.measure(0, 0)
        program, external = sampling_program([narrow, flipping_circuit()], shots=4)

        stepper = ProgramStepper(program, external)
        stepper.step()
        request, answer = self.answer(stepper)
        self.assertEqual(
            [array.shape for array in answer],
            [(4, 1), (4, 1), (4, 2)],
            "one result per register, the first circuit's before the second's",
        )
        self.assertEqual(
            [str(ty) for ty in request.output_types], ["Bit[4, 1]"] * 2 + ["Bit[4, 2]"]
        )

    def test_the_answer_satisfies_declared_types(self):
        """Test that an answer is accepted by the stepper that asked for it."""
        program, external = sampling_program([flipping_circuit()])
        stepper = ProgramStepper(program, external)
        stepper.step()
        request, answer = self.answer(stepper)
        stepper.fulfil(request.id, answer)
        stepper.step()
        self.assertIsNotNone(stepper.outputs)

    def test_an_op_that_is_not_a_shot_loop_is_refused(self):
        """Test that a function holding work other than a shot loop is refused."""
        # Declaring `multiply` alongside the shot loop puts the two in one function, since a
        # multiply reading a shot loop's outcomes sits in the same layer as it.
        outcomes = shot_loop([flipping_circuit()], shots=4)
        program = build({"doubled": outcomes[0]["a"] * 2})
        program, resources = program.partition([["qiskit.shot_loop", "qiskit.multiply"]])

        stepper = ProgramStepper(program, range(len(resources)))
        stepper.step()
        with self.assertRaisesRegex(BasicProviderError, "qiskit.multiply"):
            self.answer(stepper)


class TestRun(QiskitTestCase):
    """Tests for running a whole program."""

    def excitation_program(self, shots=4096, points=5):
        """A program sweeping `ry` over `points` angles and averaging the outcomes."""
        angles = qp_input("angles", f64[points, 1])
        outcomes = shot_loop([swept_circuit()], shots=shots, parameter_values=[angles])
        return build({"excited": outcomes[0]["c"].mean(axis=1)})

    def test_the_result_is_a_data_tree(self):
        """Test that a run gives back a data tree arranged as the program's outputs."""
        program = build({"bits": shot_loop([flipping_circuit()], shots=4)})
        result = BasicProgramSimulator(seed_simulator=1).run(program).result()
        self.assertIsInstance(result, DataTree)
        np.testing.assert_array_equal(result["bits.0.a"], np.full((4, 1), True))

    def test_post_processing_runs_in_process(self):
        """Test that arithmetic around the shot loop is evaluated by Qiskit itself."""
        outcomes = shot_loop([flipping_circuit()], shots=8)
        program = build({"mean": outcomes[0]["b"].mean(axis=0)})
        result = BasicProgramSimulator(seed_simulator=1).run(program).result()
        np.testing.assert_array_equal(result["mean"], [0.0, 1.0])

    def test_a_program_with_no_shot_loop_runs(self):
        """Test that a program needing no simulation at all is still evaluated."""
        program = build({"sum": qp_input("x", f64[2]) + qp_input("y", f64[2])})
        inputs = {"x": np.array([1.0, 2.0]), "y": np.array([10.0, 20.0])}
        np.testing.assert_array_equal(
            BasicProgramSimulator().run(program, inputs).result()["sum"], [11.0, 22.0]
        )

    def test_inputs_reach_the_shot_loop(self):
        """Test that a swept input parameter reaches the circuit it is bound into."""
        angles = np.array([[0.0], [np.pi]])
        program = self.excitation_program(shots=32, points=2)
        result = BasicProgramSimulator(seed_simulator=5).run(program, {"angles": angles})
        np.testing.assert_array_equal(result.result()["excited"], [[0.0], [1.0]])

    def test_inputs_as_a_data_tree(self):
        """Test that a data tree of inputs is taken as it is."""
        program = self.excitation_program(shots=32, points=2)
        tree = DataTree({"angles": np.array([[0.0], [np.pi]])})
        result = BasicProgramSimulator(seed_simulator=5).run(program, tree)
        np.testing.assert_array_equal(result.result()["excited"], [[0.0], [1.0]])

    def test_the_excitation_tracks_the_swept_angle(self):
        """Test that averaging a swept shot loop recovers sin(theta / 2) squared."""
        theta = np.linspace(0, np.pi, 5)
        program = self.excitation_program()
        result = BasicProgramSimulator(seed_simulator=42).run(
            program, {"angles": theta.reshape(5, 1)}
        )
        np.testing.assert_allclose(
            np.asarray(result.result()["excited"])[:, 0],
            np.sin(theta / 2) ** 2,
            atol=0.02,
        )

    def test_the_same_seed_repeats(self):
        """Test that two runs seeded alike agree."""
        program = self.excitation_program(shots=64, points=3)
        angles = np.array([[0.4], [0.8], [1.2]])
        first = BasicProgramSimulator(seed_simulator=3).run(program, {"angles": angles})
        second = BasicProgramSimulator(seed_simulator=3).run(program, {"angles": angles})
        np.testing.assert_array_equal(first.result()["excited"], second.result()["excited"])

    def test_a_simulator_repeats_across_runs(self):
        """Test that one seeded simulator gives the same answer every run."""
        program = self.excitation_program(shots=64, points=3)
        angles = np.array([[0.4], [0.8], [1.2]])
        simulator = BasicProgramSimulator(seed_simulator=3)
        np.testing.assert_array_equal(
            simulator.run(program, {"angles": angles}).result()["excited"],
            simulator.run(program, {"angles": angles}).result()["excited"],
        )

    def test_different_seeds_differ(self):
        """Test that two runs seeded apart do not agree."""
        program = self.excitation_program(shots=64, points=3)
        angles = np.array([[0.4], [0.8], [1.2]])
        first = BasicProgramSimulator(seed_simulator=3).run(program, {"angles": angles})
        second = BasicProgramSimulator(seed_simulator=4).run(program, {"angles": angles})
        self.assertFalse(
            np.array_equal(first.result()["excited"], second.result()["excited"]),
        )

    def test_a_second_loop_reads_the_first(self):
        """Test that a shot loop whose parameters come from another is run after it."""
        measured = shot_loop([flipping_circuit()], shots=8)[0]["a"]
        angles = measured.mean(axis=0) * np.pi
        outcomes = shot_loop([swept_circuit()], shots=8, parameter_values=[angles])
        program = build({"excited": outcomes[0]["c"].mean(axis=0)})

        result = BasicProgramSimulator(seed_simulator=11).run(program).result()
        # The first loop always measures a 1, so the second is swept to pi and always excited.
        np.testing.assert_array_equal(result["excited"], [1.0])

    def test_the_job_reports_done(self):
        """Test that a job exists only once its program has finished."""
        program = build({"bits": shot_loop([flipping_circuit()], shots=4)})
        job = BasicProgramSimulator(seed_simulator=1).run(program)
        self.assertIsInstance(job, BasicProgramJob)
        self.assertEqual(job.status(), JobStatus.DONE)
        self.assertIn(job.job_id, repr(job))
        self.assertIn("DONE", repr(job))

    def test_a_partitioned_program_is_refused(self):
        """Test that a program whose entry point already calls a function is refused."""
        program = build({"bits": shot_loop([flipping_circuit()], shots=4)})
        partitioned, _ = program.partition([["qiskit.shot_loop"]])
        with self.assertRaisesRegex(ValueError, "entry point calls another function"):
            BasicProgramSimulator().run(partitioned)


class TestBasicProgramRuntime(QiskitTestCase):
    """Tests for running a program in stages, suspended under a job id."""

    def swept_program(self, points=2, shots=32):
        """A program sweeping `ry` over `points` angles, and the angles to sweep."""
        angles = qp_input("angles", f64[points, 1])
        outcomes = shot_loop([swept_circuit()], shots=shots, parameter_values=[angles])
        program = build({"excited": outcomes[0]["c"].mean(axis=1)})
        return program, {"angles": np.linspace(0, np.pi, points).reshape(points, 1)}

    def test_a_run_is_fetched_by_its_id(self):
        """Test that a run continues given nothing but its job id."""
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime(seed_simulator=42)
        job_id = runtime.run(program, inputs).job_id

        # Nothing of the submitting scope is kept, matching a fresh process.
        del program, inputs
        result = runtime.job(job_id).result()
        np.testing.assert_array_equal(result["excited"], [[0.0], [1.0]])

    def test_a_fetched_job_needs_nothing_else(self):
        """Test that a fetched run finishes after the runtime it came from is gone."""
        # A fetch has to hand back everything the run consists of, so the job must not be a handle
        # that reads the service back on every call.
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime(seed_simulator=42)
        job = runtime.job(runtime.run(program, inputs).job_id)

        del runtime, program, inputs
        np.testing.assert_array_equal(job.result()["excited"], [[0.0], [1.0]])

    def test_a_job_holds_the_stepper_and_the_service_does_not(self):
        """Test that a stepper is the client's, rebuilt per fetch, and never part of a run."""
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime()
        job = runtime.run(program, inputs)

        held = {name for name in vars(job) if not name.startswith("__")}
        self.assertEqual(
            held, {"_job_id", "_program", "_stepper", "_answers", "_simulator", "_generator"}
        )
        self.assertNotIn("_runtime", held, "a job must not read the service back")

        # What a run consists of is the submission plus what has been answered, and no more.
        submitted = runtime._submitted[job.job_id]
        self.assertEqual(len(submitted), 4)
        self.assertFalse(
            any(isinstance(value, ProgramStepper) for value in submitted),
            "a service keeps no stepper, since one is replayed from the rest",
        )

    def test_a_second_fetch_replays_rather_than_reruns(self):
        """Test that a run already performed resumes without simulating any circuit again."""
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime(seed_simulator=42)
        job = runtime.run(program, inputs)
        first = job.result()

        # Every answer was recorded against the run, so replaying it needs no simulator at all.
        with mock.patch.object(BasicSimulator, "run", side_effect=AssertionError("simulated")):
            again = runtime.job(job.job_id)
            self.assertEqual(again.status(), JobStatus.DONE)
            np.testing.assert_array_equal(again.result()["excited"], first["excited"])

    def test_a_partly_answered_run_resumes_where_it_stopped(self):
        """Test that a run with one of two requests answered replays only that one."""
        outcomes = shot_loop([flipping_circuit()], shots=8)
        swept = shot_loop([swept_circuit()], shots=8, parameter_values=[qp_input("angles", f64[1])])
        program = build({"a": outcomes[0]["a"], "c": swept[0]["c"]})

        runtime = BasicProgramRuntime(seed_simulator=5)
        job_id = runtime.run(program, {"angles": np.array([np.pi])}).job_id
        held, external, inputs, answers = runtime._submitted[job_id]

        # Record an answer to one of the two requests, as a service would when a client reports it.
        stepper = replay(held, external, inputs, answers)
        requests = stepper.outstanding()
        self.assertEqual(len(requests), 2)
        answers[requests[0].id] = answer_request(
            requests[0], BasicSimulator(), np.random.default_rng(5)
        )

        # Fetching now replays that one and leaves the other outstanding.
        resumed = runtime.job(job_id)
        self.assertEqual(resumed.status(), JobStatus.RUNNING)
        self.assertEqual(len(resumed._stepper.outstanding()), 1)
        np.testing.assert_array_equal(resumed.result()["c"], np.ones((8, 1), dtype=bool))

    def test_a_run_agrees_with_the_simulator(self):
        """Test that suspending a run changes nothing about its answer."""
        program, inputs = self.swept_program(points=3, shots=64)
        staged = BasicProgramRuntime(seed_simulator=3).run(program, inputs).result()
        whole = BasicProgramSimulator(seed_simulator=3).run(program, inputs).result()
        np.testing.assert_array_equal(staged["excited"], whole["excited"])

    def test_a_run_of_two_stages_resumes_through_both(self):
        """Test that a program whose second shot loop reads the first is carried through."""
        measured = shot_loop([flipping_circuit()], shots=8)[0]["a"]
        outcomes = shot_loop(
            [swept_circuit()], shots=8, parameter_values=[measured.mean(axis=0) * np.pi]
        )
        program = build({"excited": outcomes[0]["c"].mean(axis=0)})

        job = BasicProgramRuntime(seed_simulator=11).run(program)
        self.assertEqual(job.status(), JobStatus.RUNNING)
        np.testing.assert_array_equal(job.result()["excited"], [1.0])

    def test_a_program_needing_nothing_is_done_at_once(self):
        """Test that a program with no shot loop is finished by the step that submits it."""
        program = build({"sum": qp_input("x", f64[2]) + qp_input("y", f64[2])})
        inputs = {"x": np.array([1.0, 2.0]), "y": np.array([10.0, 20.0])}
        job = BasicProgramRuntime().run(program, inputs)
        self.assertEqual(job.status(), JobStatus.DONE)
        np.testing.assert_array_equal(job.result()["sum"], [11.0, 22.0])

    def test_a_finished_run_is_fetched_again(self):
        """Test that a run already finished reports its result without redoing the work."""
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime(seed_simulator=1)
        job = runtime.run(program, inputs)
        first = job.result()

        again = runtime.job(job.job_id)
        self.assertEqual(again.status(), JobStatus.DONE)
        np.testing.assert_array_equal(again.result()["excited"], first["excited"])

    def test_an_unknown_job_id_is_refused(self):
        """Test that asking for a run the runtime does not hold is refused."""
        with self.assertRaisesRegex(BasicProviderError, "no run was submitted"):
            BasicProgramRuntime().job("nonesuch")

    def test_the_repr_reports_what_is_held(self):
        """Test that the reprs name the job and how many runs are held."""
        program, inputs = self.swept_program()
        runtime = BasicProgramRuntime(seed_simulator=1)
        job = runtime.run(program, inputs)
        self.assertEqual(repr(job), f"BasicRuntimeJob({job.job_id}, RUNNING)")
        self.assertIn("1 run(s) submitted", repr(runtime))
