from qiskit.transpiler import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import Instruction
from fractions import Fraction
from monodromy.coordinates import unitary_to_monodromy_coordinate
from monodromy.coverage import deduce_qlr_consequences
from monodromy.static.examples import exactly, identity_polytope, \
    everything_polytope
from monodromy.coverage import build_coverage_set, CircuitPolytope, print_coverage_set
import logging
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
import retworkx

class MonodromyDepth(AnalysisPass):
    """
    MonodromyDepth class extends the AnalysisPass to perform cost analysis on a given 
    CircuitDAG with respect to a specified 2-qubit basis gate. This basis gate is crucial in 
    calculating the minimum execution cost of 2-qubit blocks within the CircuitDAG.

    This class is particularly useful for quantum circuit optimization where the cost 
    associated with the execution of certain gates is a crucial factor in the overall performance 
    of the quantum computer.

    This class requires the Collect2qBlocks and ConsolidateBlocks passes to decompose the 
    CircuitDAG into 2-qubit blocks and consolidate them respectively. 
    """

    def __init__(self, basis_gate: Instruction):
        """
        Constructor takes a qiskit.Instruction for the basis gate.
        In the future, this can be extended to accept a list of basis gates.
        
        :param basis_gate: A unitary representing the basis gate for the analysis.
        """
        super().__init__()
        assert basis_gate.num_qubits == 2, "Basis gate must be a 2Q gate."
        self.requires = [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        self.basis_gate = basis_gate
        self.chatty = False
        self.coverage_set = self._gate_set_to_coverage()
    
    def _operation_to_circuit_polytope(self, operation: Instruction) -> CircuitPolytope:
        """
        The operation_to_circuit_polytope() function takes a qiskit.Instruction object and returns a 
        CircuitPolytope object that represents the unitary of the operation.

        Reference: https://github.com/evmckinney9/monodromy/blob/main/scripts/single_circuit.py

        :param operation: A qiskit.Instruction object.
        :return: A CircuitPolytope object
        """

        gd = operation.to_matrix()
        b_polytope = exactly(
            *(
                Fraction(x).limit_denominator(10_000)
                for x in unitary_to_monodromy_coordinate(gd)[:-1]
            )
        )
        convex_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=identity_polytope,
            b_polytope=b_polytope,
            c_polytope=everything_polytope,
        )

        return CircuitPolytope(
            operations=[operation.name],
            cost=1,
            convex_subpolytopes=convex_polytope.convex_subpolytopes,
        )
    
    def _gate_set_to_coverage(self):
        """
        The gate_set_to_coverage() function takes the basis gate and creates a CircuitPolytope object 
        that represents all the possible 2Q unitaries that can be formed by piecing together different 
        instances of the basis gate.

        :return: A CircuitPolytope object
        """
        if self.chatty:
            logging.info("==== Working to build a set of covering polytopes ====")

        # TODO, here could add functionality for multiple basis gates
        # just need to fix the cost function to account for relative durations
        operations = [self._operation_to_circuit_polytope(self.basis_gate)]
        coverage_set = build_coverage_set(operations, chatty=self.chatty)

        # TODO: add some warning or fail condition if the coverage set fails to coverage
        # one way, (but there may be a more direct way) is to check if expected haar == 0

        if self.chatty:
            logging.info("==== Done. Here's what we found: ====")
            logging.info(print_coverage_set(coverage_set))

        sorted_polytopes = sorted(coverage_set, key=lambda k: k.cost)
        return sorted_polytopes

    def _operation_to_cost(self, operation: Instruction) -> int:
        target_coords = unitary_to_monodromy_coordinate(operation.to_matrix())
        for i, circuit_polytope in enumerate(self.coverage_set):
            if circuit_polytope.has_element(target_coords):
                return i
        raise TranspilerError("Operation not found in coverage set.")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        The run() method is the main entry point for the AnalysisPass. It takes a CircuitDAG as input 
        and returns an updated CircuitDAG. This method applies the basis gate to the CircuitDAG, 
        computes the cost of the applied gate, and updates the CircuitDAG accordingly.

        :param dag: The CircuitDAG to be analyzed.
        :return: An updated CircuitDAG.
        """
        def weight_fn(_1, node, _2):
            """Weight function for longest path algorithm"""
            target_node = dag._multi_graph[node]
            if not isinstance(target_node, DAGOpNode):
                return 0
            elif target_node.op.name in ["barrier", "measure"]:
                return 0
            elif len(target_node.qargs) == 1:
                return 0
            elif len(target_node.qargs) > 2:
                raise TranspilerError("Operation not supported.")
            else:
                return self._operation_to_cost(target_node.op)
        
        longest_path_length = retworkx.dag_longest_path_length(dag._multi_graph, weight_fn=weight_fn)
        if self.chatty:
            logging.info(f"Longest path length: {longest_path_length}")
        
        self.property_set["monodromy_depth"] = longest_path_length
        return dag
