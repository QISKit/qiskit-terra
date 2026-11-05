# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass for Hoare logic circuit optimization."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals


@_optionals.HAS_Z3.require_in_instance
class HoareOptimizer(TransformationPass):
    """This is a transpiler pass using Hoare logic circuit optimization.
    The inner workings of this are detailed in:
    https://arxiv.org/abs/1810.00375
    """

    def __init__(self, size=10):
        """
        Args:
            size (int): size of gate cache, in number of gates
        Raises:
            MissingOptionalLibraryError: if unable to import z3 solver
        """
        # This module is just a script that adds several post conditions onto existing classes.
        from . import _gate_extension  # noqa: F401

        super().__init__()
        self.solver = None
        self.variables = None
        self.gatenum = None
        self.gatecache = None
        self.varnum = None
        self.size = size

    def _gen_variable(self, qubit):
        """After each gate generate a new unique variable name for each of the
            qubits, using scheme: 'q[id]_[gatenum]', e.g. q1_0 -> q1_1 -> q1_2,
                                                          q2_0 -> q2_1
        Args:
            qubit (Qubit): qubit to generate new variable for
        Returns:
            BoolRef: z3 variable of qubit state
        """
        import z3

        varname = "q" + str(qubit) + "_" + str(self.gatenum[qubit])
        var = z3.Bool(varname)
        self.gatenum[qubit] += 1
        self.variables[qubit].append(var)
        return var

    def _initialize(self, dag):
        """create boolean variables for each qubit and apply qb == 0 condition
        Args:
            dag (DAGCircuit): input DAG to get qubits from
        """
        import z3

        for qbt in dag.qubits:
            self.gatenum[qbt] = 0
            self.variables[qbt] = []
            self.gatecache[qbt] = []
            self.varnum[qbt] = {}
            x = self._gen_variable(qbt)
            self.solver.add(z3.Not(x))

    def _add_postconditions(self, gate, ctrl_ones, trgtqb, trgtvar):
        """create boolean variables for each qubit the gate is applied to
            and apply the relevant post conditions.
            a gate rotating out of the z-basis will not have any valid
            post-conditions, in which case the qubit state is unknown
        Args:
            gate (Gate): gate to inspect
            ctrl_ones (BoolRef): z3 condition asserting all control qubits to 1
            trgtqb (list((QuantumRegister, int))): list of target qubits
            trgtvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of target qubits
        """
        import z3

        new_vars = []
        for qbt in trgtqb:
            new_vars.append(self._gen_variable(qbt))

        try:
            self.solver.add(z3.Implies(ctrl_ones, gate._postconditions(*(trgtvar + new_vars))))
        except AttributeError:
            pass

        for i, tvar in enumerate(trgtvar):
            self.solver.add(z3.Implies(z3.Not(ctrl_ones), new_vars[i] == tvar))

    def _test_gate(self, gate, ctrl_ones, trgtvar):
        """use z3 sat solver to determine triviality of gate
        Args:
            gate (Gate): gate to inspect
            ctrl_ones (BoolRef): z3 condition asserting all control qubits to 1
            trgtvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of target qubits
        Returns:
            bool: if gate is trivial
        """
        import z3

        trivial = False
        self.solver.push()

        try:
            triv_cond = gate._trivial_if(*trgtvar)
        except AttributeError:
            self.solver.add(ctrl_ones)
            trivial = self.solver.check() == z3.unsat
        else:
            if isinstance(triv_cond, bool):
                if triv_cond and len(trgtvar) == 1:
                    self.solver.add(z3.Not(z3.And(ctrl_ones, trgtvar[0])))
                    sol1 = self.solver.check() == z3.unsat
                    self.solver.pop()
                    self.solver.push()
                    self.solver.add(z3.And(ctrl_ones, trgtvar[0]))
                    sol2 = self.solver.check() == z3.unsat
                    trivial = sol1 or sol2
            else:
                self.solver.add(z3.And(ctrl_ones, z3.Not(triv_cond)))
                trivial = self.solver.check() == z3.unsat

        self.solver.pop()
        return trivial

    def _remove_control(self, gate, ctrlvar, trgtvar):
        """use z3 sat solver to determine if all control qubits are in 1 state,
             and if so replace the Controlled - U by U.
        Args:
            gate (Gate): gate to inspect
            ctrlvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of control qubits
            trgtvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of target qubits
        Returns:
            Tuple(bool, DAGCircuit, List)::
              * bool:if controlled gate can be replaced.
              * DAGCircuit: with U applied to the target qubits.
              * List: with indices of target qubits.
        """
        remove = False

        qarg = QuantumRegister(gate.num_qubits)
        dag = DAGCircuit()
        dag.add_qreg(qarg)

        qb = list(range(len(ctrlvar), gate.num_qubits))  # last qubits as target.

        if isinstance(gate, ControlledGate):
            remove = self._check_removal(ctrlvar)

        # try with other qubit as 'target'.
        if isinstance(gate, (CZGate, CU1Gate, MCU1Gate)):
            while not remove and qb[0] > 0:
                qb[0] = qb[0] - 1
                ctrl_vars = ctrlvar[: qb[0]] + ctrlvar[qb[0] + 1 :] + trgtvar
                remove = self._check_removal(ctrl_vars)

        if remove:
            qubits = [qarg[qi] for qi in qb]
            dag.apply_operation_back(gate.base_gate, qubits)

        return remove, dag, qb

    def _check_removal(self, ctrlvar):
        import z3

        ctrl_ones = z3.And(*ctrlvar)

        self.solver.push()
        self.solver.add(z3.Not(ctrl_ones))
        remove = self.solver.check() == z3.unsat
        self.solver.pop()

        return remove

    def _traverse_dag(self, dag):
        """traverse DAG in topological order
            for each gate check: if any control is 0, or
                                 if triviality conditions are satisfied
            if yes remove gate from dag
            apply postconditions of gate
        Args:
            dag (DAGCircuit): input DAG to optimize in place
        """
        import z3

        # Pre-generate all DAG nodes, since we later iterate over them, while
        # potentially modifying and removing some of them.
        nodes = list(dag.topological_op_nodes())
        for node in nodes:
            gate = node.op
            _, ctrlvar, trgtqb, trgtvar = self._separate_ctrl_trgt(node)

            ctrl_ones = z3.And(*ctrlvar)

            remove_ctrl, new_dag, _ = self._remove_control(gate, ctrlvar, trgtvar)

            if remove_ctrl:
                # We are replacing a node by a new node over a smaller number of qubits.
                # This can be done using substitute_node_with_dag, which furthermore returns
                # a mapping from old node ids to new nodes.
                mapped_nodes = dag.substitute_node_with_dag(node, new_dag)
                node = next(iter(mapped_nodes.values()))
                gate = node.op
                _, ctrlvar, trgtqb, trgtvar = self._separate_ctrl_trgt(node)

                ctrl_ones = z3.And(*ctrlvar)

            trivial = self._test_gate(gate, ctrl_ones, trgtvar)
            if trivial:
                dag.remove_op_node(node)
            elif self.size > 1:
                for qbt in node.qargs:
                    self.gatecache[qbt].append(node)
                    self.varnum[qbt][node] = self.gatenum[qbt] - 1
                for qbt in node.qargs:
                    if len(self.gatecache[qbt]) >= self.size:
                        self._multigate_opt(dag, qbt)

            self._add_postconditions(gate, ctrl_ones, trgtqb, trgtvar)

    def _remove_successive_identity(self, dag, qubit, from_idx=None):
        """remove gates that have the same set of target qubits, follow each
            other immediately on these target qubits, and combine to the
            identity (consider sequences of length 2 for now)
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
            qubit (Qubit): qubit cache to inspect
            from_idx (int): only gates whose indexes in the cache are larger
                            than this value can be removed
        """
        i = 0
        while i < len(self.gatecache[qubit]) - 1:
            append = True
            node1 = self.gatecache[qubit][i]
            node2 = self.gatecache[qubit][i + 1]
            trgtqb1 = self._separate_ctrl_trgt(node1)[2]
            trgtqb2 = self._separate_ctrl_trgt(node2)[2]
            i += 1

            if trgtqb1 != trgtqb2:
                continue
            try:
                for qbt in trgtqb1:
                    idx = self.gatecache[qbt].index(node1)
                    if self.gatecache[qbt][idx + 1] is not node2:
                        append = False
            except (IndexError, ValueError):
                continue

            seq = [node1, node2]
            if append and self._is_identity(seq) and self._seq_as_one(seq):
                i += 1
                for node in seq:
                    dag.remove_op_node(node)
                    if from_idx is None or self.gatecache[qubit].index(node) > from_idx:
                        for qbt in node.qargs:
                            self.gatecache[qbt].remove(node)

    def _is_identity(self, sequence):
        """determine whether the sequence of gates combines to the identity
            (consider sequences of length 2 for now)
        Args:
            sequence (list(DAGOpNode)): gate sequence to inspect
        Returns:
            bool: if gate sequence combines to identity
        """
        if len(sequence) != 2:
            raise ValueError("Invalid sequence length")
        # some Instructions (e.g measurements) may not have an inverse.
        try:
            gate1, gate2 = sequence[0].op, sequence[1].op.inverse()
        except CircuitError:
            return False
        par1, par2 = gate1.params, gate2.params
        def1, def2 = gate1.definition, gate2.definition

        if isinstance(gate1, ControlledGate):
            gate1 = gate1.base_gate
        gate1 = gate1.base_class
        if isinstance(gate2, ControlledGate):
            gate2 = gate2.base_gate
        gate2 = gate2.base_class

        # equality of gates can be determined via type and parameters, unless
        # the gates have no specific type, in which case definition is used
        # or they are unitary gates, in which case matrix equality is used
        if gate1 is Gate and gate2 is Gate:
            return def1 == def2 and def1 and def2
        elif gate1 is UnitaryGate and gate2 is UnitaryGate:
            return matrix_equal(par1[0], par2[0], ignore_phase=True)

        return gate1 == gate2 and par1 == par2

    def _seq_as_one(self, sequence):
        """use z3 solver to determine if the gates in the sequence are either
            all executed or none of them are executed, based on control qubits
            (consider sequences of length 2 for now)
        Args:
            sequence (list(DAGOpNode)): gate sequence to inspect
        Returns:
            bool: if gate sequence is only executed completely or not at all
        """
        from z3 import Or, And, Not
        import z3

        if len(sequence) != 2:
            raise ValueError("Invalid sequence length")
        ctrlvar1 = self._separate_ctrl_trgt(sequence[0])[1]
        ctrlvar2 = self._separate_ctrl_trgt(sequence[1])[1]

        self.solver.push()
        self.solver.add(
            Or(And(And(*ctrlvar1), Not(And(*ctrlvar2))), And(Not(And(*ctrlvar1)), And(*ctrlvar2)))
        )
        res = self.solver.check() == z3.unsat
        self.solver.pop()

        return res

    def _multigate_opt(self, dag, qubit, max_idx=None, dnt_rec=None):
        """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
            qubit (Qubit): qubit whose gate cache is to be optimized
            max_idx (int): a value indicates a recursive call, optimize
                           and remove gates up to this point in the cache
            dnt_rec (list(int)): don't recurse on these qubit caches (again)
        """
        if not self.gatecache[qubit]:
            return

        # try to optimize this qubit's pipeline
        self._remove_successive_identity(dag, qubit, max_idx)
        if len(self.gatecache[qubit]) < self.size and max_idx is None:
            # unless in a rec call, we are done if the cache isn't full
            return
        elif max_idx is None:
            # need to remove at least one gate from cache, so remove oldest
            max_idx = 0
            dnt_rec = set()
            dnt_rec.add(qubit)
            gates_tbr = [self.gatecache[qubit][0]]
        else:
            # need to remove all gates up to max_idx (in reverse order)
            gates_tbr = self.gatecache[qubit][max_idx::-1]

        for node in gates_tbr:
            # for rec call, only look at qubits that haven't been optimized yet
            new_qb = [x for x in node.qargs if x not in dnt_rec]
            dnt_rec.update(new_qb)
            for qbt in new_qb:
                idx = self.gatecache[qbt].index(node)
                # recursive chain to optimize all gates in this qubit's cache
                self._multigate_opt(dag, qbt, max_idx=idx, dnt_rec=dnt_rec)
        # truncate gatecache for this qubit to after above gate
        self.gatecache[qubit] = self.gatecache[qubit][max_idx + 1 :]

    def _separate_ctrl_trgt(self, node):
        """Get the target qubits and control qubits if available,
        as well as their respective z3 variables.
        """
        gate = node.op
        if isinstance(gate, ControlledGate):
            numctrl = gate.num_ctrl_qubits
        else:
            numctrl = 0
        ctrlqb = node.qargs[:numctrl]
        trgtqb = node.qargs[numctrl:]
        try:
            ctrlvar = [self.variables[qb][self.varnum[qb][node]] for qb in ctrlqb]
            trgtvar = [self.variables[qb][self.varnum[qb][node]] for qb in trgtqb]
        except KeyError:
            ctrlvar = [self.variables[qb][-1] for qb in ctrlqb]
            trgtvar = [self.variables[qb][-1] for qb in trgtqb]
        return (ctrlqb, ctrlvar, trgtqb, trgtvar)

    def _reset(self):
        """Reset HoareOptimize internal state,
        so it can be run multiple times.
        """
        import z3

        self.solver = z3.Solver()
        self.variables = {}
        self.gatenum = {}
        self.gatecache = {}
        self.varnum = {}

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        self._reset()
        self._initialize(dag)
        self._traverse_dag(dag)
        if self.size > 1:
            for qbt in dag.qubits:
                self._multigate_opt(dag, qbt)
        return dag
