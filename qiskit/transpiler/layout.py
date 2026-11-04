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

"""
A two-ways dict to represent a layout.

Layout is the relation between virtual (qu)bits and physical (qu)bits.
Virtual (qu)bits are tuples, e.g. `(QuantumRegister(3, 'qr'), 2)` or simply `qr[2]`.
Physical (qu)bits are integers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from qiskit import circuit
from qiskit.circuit import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.transpiler import PropertySet


class Layout:
    """Two-ways dict to represent a Layout."""

    __slots__ = ("_p2v", "_regs", "_v2p")

    def __init__(self, input_dict=None):
        """construct a Layout from a bijective dictionary, mapping
        virtual qubits to physical qubits"""
        self._regs = []
        self._p2v = {}
        self._v2p = {}
        if input_dict is not None:
            if not isinstance(input_dict, dict):
                raise LayoutError("Layout constructor takes a dict")
            self.from_dict(input_dict)

    def __repr__(self):
        """Representation of a Layout"""
        str_list = []
        for key, val in self._p2v.items():
            str_list.append(f"{key}: {val},")
        if str_list:
            str_list[-1] = str_list[-1][:-1]
        return "Layout({\n" + "\n".join(str_list) + "\n})"

    def from_dict(self, input_dict):
        """Populates a Layout from a dictionary.

        The dictionary must be a bijective mapping between
        virtual qubits (tuple) and physical qubits (int).

        Args:
            input_dict (dict):
                e.g.::

                {(QuantumRegister(3, 'qr'), 0): 0,
                 (QuantumRegister(3, 'qr'), 1): 1,
                 (QuantumRegister(3, 'qr'), 2): 2}

                Can be written more concisely as follows:

                * virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 1,
                     qr[2]: 2}

                * physical to virtual::

                    {0: qr[0],
                     1: qr[1],
                     2: qr[2]}
        """
        for key, value in input_dict.items():
            virtual, physical = Layout.order_based_on_type(key, value)
            self._p2v[physical] = virtual
            if virtual is None:
                continue
            self._v2p[virtual] = physical

    @staticmethod
    def order_based_on_type(value1, value2):
        """decides which one is physical/virtual based on the type. Returns (virtual, physical)"""
        if isinstanceint(value1) and isinstance(value2, (Qubit, type(None))):
            physical = int(value1)
            virtual = value2
        elif isinstanceint(value2) and isinstance(value1, (Qubit, type(None))):
            physical = int(value2)
            virtual = value1
        else:
            raise LayoutError(
                f"The map ({type(value1)} -> {type(value2)}) has to be a (Bit -> integer)"
                " or the other way around."
            )
        return virtual, physical

    def __getitem__(self, item):
        if item in self._p2v:
            return self._p2v[item]
        if item in self._v2p:
            return self._v2p[item]
        raise KeyError(f"The item {item} does not exist in the Layout")

    def __contains__(self, item):
        return item in self._p2v or item in self._v2p

    def __setitem__(self, key, value):
        virtual, physical = Layout.order_based_on_type(key, value)
        self._set_type_checked_item(virtual, physical)

    def _set_type_checked_item(self, virtual, physical):
        old = self._v2p.pop(virtual, None)
        self._p2v.pop(old, None)
        old = self._p2v.pop(physical, None)
        self._v2p.pop(old, None)

        self._p2v[physical] = virtual
        if virtual is not None:
            self._v2p[virtual] = physical

    def __delitem__(self, key):
        if isinstance(key, int):
            del self._v2p[self._p2v[key]]
            del self._p2v[key]
        elif isinstance(key, Qubit):
            del self._p2v[self._v2p[key]]
            del self._v2p[key]
        else:
            raise LayoutError(
                "The key to remove should be of the form"
                f" Qubit or integer) and {type(key)} was provided"
            )

    def __len__(self):
        return len(self._p2v)

    def __eq__(self, other):
        if isinstance(other, Layout):
            return self._p2v == other._p2v and self._v2p == other._v2p
        return False

    def copy(self):
        """Returns a copy of a Layout instance."""
        layout_copy = type(self)()

        layout_copy._regs = self._regs.copy()
        layout_copy._p2v = self._p2v.copy()
        layout_copy._v2p = self._v2p.copy()

        return layout_copy

    def add(self, virtual_bit, physical_bit=None):
        """
        Adds a map element between `bit` and `physical_bit`. If `physical_bit` is not
        defined, `bit` will be mapped to a new physical bit.

        Args:
            virtual_bit (tuple): A (qu)bit. For example, (QuantumRegister(3, 'qr'), 2).
            physical_bit (int): A physical bit. For example, 3.
        """
        if physical_bit is None:
            if len(self._p2v) == 0:
                physical_bit = 0
            else:
                max_physical = max(self._p2v)
                # Fill any gaps in the existing bits
                for physical_candidate in range(max_physical):
                    if physical_candidate not in self._p2v:
                        physical_bit = physical_candidate
                        break
                # If there are no free bits in the allocated physical bits add new ones
                else:
                    physical_bit = max_physical + 1

        self[virtual_bit] = physical_bit

    def add_register(self, reg):
        """Adds at the end physical_qubits that map each bit in reg.

        Args:
            reg (Register): A (qu)bit Register. For example, QuantumRegister(3, 'qr').
        """
        self._regs.append(reg)
        for bit in reg:
            if bit not in self:
                self.add(bit)

    def get_registers(self):
        """
        Returns the registers in the layout [QuantumRegister(2, 'qr0'), QuantumRegister(3, 'qr1')]
        Returns:
            Set: A set of Registers in the layout
        """
        return set(self._regs)

    def get_virtual_bits(self):
        """
        Returns the dictionary where the keys are virtual (qu)bits and the
        values are physical (qu)bits.
        """
        return self._v2p

    def get_physical_bits(self):
        """
        Returns the dictionary where the keys are physical (qu)bits and the
        values are virtual (qu)bits.
        """
        return self._p2v

    def swap(self, left, right):
        """Swaps the map between left and right.

        Args:
            left (tuple or int): Item to swap with right.
            right (tuple or int): Item to swap with left.
        Raises:
            LayoutError: If left and right have not the same type.
        """
        if type(left) is not type(right):
            raise LayoutError("The method swap only works with elements of the same type.")
        temp = self[left]
        self[left] = self[right]
        self[right] = temp

    def combine_into_edge_map(self, another_layout):
        """Combines self and another_layout into an "edge map".

        For example::

              self       another_layout  resulting edge map
           qr_1 -> 0        0 <- q_2         qr_1 -> q_2
           qr_2 -> 2        2 <- q_1         qr_2 -> q_1
           qr_3 -> 3        3 <- q_0         qr_3 -> q_0

        The edge map is used to compose dags via, for example, compose.

        Args:
            another_layout (Layout): The other layout to combine.
        Returns:
            dict: A "edge map".
        Raises:
            LayoutError: another_layout can be bigger than self, but not smaller.
                Otherwise, raises.
        """
        edge_map = {}

        for virtual, physical in self._v2p.items():
            if physical not in another_layout._p2v:
                raise LayoutError(
                    "The wire_map_from_layouts() method does not support when the"
                    " other layout (another_layout) is smaller."
                )
            edge_map[virtual] = another_layout[physical]

        return edge_map

    def reorder_bits(self, bits) -> list[int]:
        """Given an ordered list of bits, reorder them according to this layout.

        The list of bits must exactly match the virtual bits in this layout.

        Args:
            bits (list[Bit]): the bits to reorder.

        Returns:
            List: ordered bits.
        """
        order = [0] * len(bits)

        # the i-th bit is now sitting in position j
        for i, v in enumerate(bits):
            j = self[v]
            order[i] = j

        return order

    @staticmethod
    def generate_trivial_layout(*regs):
        """Creates a trivial ("one-to-one") Layout with the registers and qubits in `regs`.

        Args:
            *regs (Registers, Qubits): registers and qubits to include in the layout.
        Returns:
            Layout: A layout with all the `regs` in the given order.
        """
        layout = Layout()
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                layout.add(reg)
        return layout

    @staticmethod
    def from_intlist(int_list, *qregs):
        """Converts a list of integers to a Layout
        mapping virtual qubits (index of the list) to
        physical qubits (the list values).

        Args:
            int_list (list): A list of integers.
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: The corresponding Layout object.
        Raises:
            LayoutError: Invalid input layout.
        """
        if not all(isinstanceint(i) for i in int_list):
            raise LayoutError("Expected a list of ints")
        if len(int_list) != len(set(int_list)):
            raise LayoutError("Duplicate values not permitted; Layout is bijective.")
        num_qubits = sum(reg.size for reg in qregs)
        # Check if list is too short to cover all qubits
        if len(int_list) != num_qubits:
            raise LayoutError(
                f"Integer list length ({len(int_list)}) must equal number of qubits "
                f"in circuit ({num_qubits}): {int_list}."
            )
        out = Layout()
        main_idx = 0
        for qreg in qregs:
            for idx in range(qreg.size):
                out[qreg[idx]] = int_list[main_idx]
                main_idx += 1
            out.add_register(qreg)
        if main_idx != len(int_list):
            for int_item in int_list[main_idx:]:
                out[int_item] = None
        return out

    @staticmethod
    def from_qubit_list(qubit_list, *qregs):
        """
        Populates a Layout from a list containing virtual
        qubits, Qubit or None.

        Args:
            qubit_list (list):
                e.g.: [qr[0], None, qr[2], qr[3]]
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: the corresponding Layout object
        Raises:
            LayoutError: If the elements are not Qubit or None
        """
        out = Layout()
        for physical, virtual in enumerate(qubit_list):
            if virtual is None:
                continue
            if isinstance(virtual, Qubit):
                if virtual in out._v2p:
                    raise LayoutError("Duplicate values not permitted; Layout is bijective.")
                out[virtual] = physical
            else:
                raise LayoutError("The list should contain elements of the Bits or NoneTypes")
        for qreg in qregs:
            out.add_register(qreg)
        return out

    def compose(self, other: Layout, qubits: list[Qubit]) -> Layout:
        """Compose this layout with another layout.

        If this layout represents a mapping from the P-qubits to the positions of the Q-qubits,
        and the other layout represents a mapping from the Q-qubits to the positions of
        the R-qubits, then the composed layout represents a mapping from the P-qubits to the
        positions of the R-qubits.

        Args:
            other: The existing :class:`.Layout` to compose this :class:`.Layout` with.
            qubits: A list of :class:`.Qubit` objects over which ``other`` is defined,
                used to establish the correspondence between the positions of the ``other``
                qubits and the actual qubits.

        Returns:
            A new layout object the represents this layout composed with the ``other`` layout.
        """
        other_v2p = other.get_virtual_bits()
        return Layout({virt: other_v2p[qubits[phys]] for virt, phys in self._v2p.items()})

    def inverse(self, source_qubits: list[Qubit], target_qubits: list[Qubit]):
        """Finds the inverse of this layout.

        This is possible when the layout is a bijective mapping, however the input
        and the output qubits may be different (in particular, this layout may be
        the mapping from the extended-with-ancillas virtual qubits to physical qubits).
        Thus, if this layout represents a mapping from the P-qubits to the positions
        of the Q-qubits, the inverse layout represents a mapping from the Q-qubits
        to the positions of the P-qubits.

        Args:
            source_qubits: A list of :class:`.Qubit` objects representing the domain
                of the layout.
            target_qubits: A list of :class:`.Qubit` objects representing the image
                of the layout.

        Returns:
            A new layout object the represents the inverse of this layout.
        """
        source_qubit_to_position = {q: p for p, q in enumerate(source_qubits)}
        return Layout(
            {
                target_qubits[pos_phys]: source_qubit_to_position[virt]
                for virt, pos_phys in self._v2p.items()
            }
        )

    def to_permutation(self, qubits: list[Qubit]):
        """Creates a permutation corresponding to this layout.

        This is possible when the layout is a bijective mapping with the same
        source and target qubits (for instance, a "final_layout" corresponds
        to a permutation of the physical circuit qubits). If this layout is
        a mapping from qubits to their new positions, the resulting permutation
        describes which qubits occupy the positions 0, 1, 2, etc. after
        applying the permutation.

        For example, suppose that the list of qubits is ``[qr_0, qr_1, qr_2]``,
        and the layout maps ``qr_0`` to ``2``, ``qr_1`` to ``0``, and
        ``qr_2`` to ``1``. In terms of positions in ``qubits``, this maps ``0``
        to ``2``, ``1`` to ``0`` and ``2`` to ``1``, with the corresponding
        permutation being ``[1, 2, 0]``.
        """

        perm = [None] * len(qubits)
        for i, q in enumerate(qubits):
            pos = self._v2p[q]
            perm[pos] = i
        return perm


@dataclass
class TranspileLayout:
    r"""Layout attributes for the output circuit from transpiler.

    The :mod:`~qiskit.transpiler` is unitary-preserving up to the "initial layout"
    and "final layout" permutations. The initial layout permutation is caused by
    setting and applying the initial layout during the :ref:`transpiler-preset-stage-layout`.
    The final layout permutation is caused by :class:`~.SwapGate` insertion during
    the :ref:`transpiler-preset-stage-routing`. This class provides an interface to reason about
    these permutations using a variety of helper methods.

    During the layout stage, the transpiler can potentially remap the order of the
    qubits in the circuit as it fits the circuit to the target backend. For example,
    let the input circuit be:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit import QuantumCircuit, QuantumRegister

       qr = QuantumRegister(3, name="MyReg")
       qc = QuantumCircuit(qr)
       qc.h(0)
       qc.cx(0, 1)
       qc.cx(0, 2)
       qc.draw("mpl")


    Suppose that during the layout stage the transpiler reorders the qubits to be:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.cx(2, 0)
       qc.draw("mpl")

    Then the output of the :meth:`.initial_virtual_layout` method is
    equivalent to::

        Layout({
            qr[0]: 2,
            qr[1]: 1,
            qr[2]: 0,
        })

    (it is also this attribute in the :meth:`.QuantumCircuit.draw` and
    :func:`.circuit_drawer` which is used to display the mapping of qubits to
    positions in circuit visualizations post-transpilation).

    Building on the above example, suppose that during the routing stage
    the transpiler needs to insert swap gates, and the output circuit
    becomes:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.swap(0, 1)
       qc.cx(2, 1)
       qc.draw("mpl")

    Then the output of the :meth:`routing_permutation` method is::

        [1, 0, 2]

    which maps positions of qubits before routing to their final positions
    after routing.

    There are three public attributes associated with the class, however these
    are mostly provided for backwards compatibility and represent the internal
    state from the transpiler. They are defined as:

      * :attr:`initial_layout` - This attribute is used to model the
        permutation caused by the :ref:`transpiler-preset-stage-layout`. It is a
        :class:`~.Layout` object that maps the input :class:`~.QuantumCircuit`\s
        :class:`~.circuit.Qubit` objects to the position in the output
        :class:`.QuantumCircuit.qubits` list.
      * :attr:`input_qubit_mapping` - This attribute is used to retain
        input ordering of the original :class:`~.QuantumCircuit` object. It
        maps the virtual :class:`~.circuit.Qubit` object from the original circuit
        (and :attr:`initial_layout`) to its corresponding position in
        :attr:`.QuantumCircuit.qubits` in the original circuit. This
        is needed when computing the permutation of the :class:`Operator` of
        the circuit (and used by :meth:`.Operator.from_circuit`).
      * :attr:`final_layout` - This attribute is used to model the
        permutation caused by the :ref:`transpiler-preset-stage-routing`. It is a
        :class:`~.Layout` object that maps the output circuit's qubits from
        :class:`.QuantumCircuit.qubits` in the output circuit to their final
        positions after routing. Importantly, this only represents the
        permutation caused by inserting :class:`~.SwapGate`\s into
        the :class:`~.QuantumCircuit` during the :ref:`transpiler-preset-stage-routing`.
        It is **not** a mapping from the original input circuit's position
        to the final position at the end of the transpiled circuit.
        If you need this, you can use the :meth:`.final_index_layout` to generate this.
        If :attr:`final_layout` is set to ``None``, this indicates that routing was not
        run, and can be considered equivalent to a trivial layout with the qubits from
        the output circuit's :attr:`~.QuantumCircuit.qubits` list.
    """

    initial_layout: Layout
    input_qubit_mapping: dict[circuit.Qubit, int]
    final_layout: Layout | None = None
    _input_qubit_count: int | None = None
    _output_qubit_list: list[Qubit] | None = None

    def initial_virtual_layout(self, filter_ancillas: bool = False) -> Layout:
        """Return a :class:`.Layout` object for the initial layout.

        This returns a mapping of virtual :class:`~.circuit.Qubit` objects in the input
        circuit to the positions of the physical qubits selected during layout.
        This is analogous to the :attr:`.initial_layout` attribute.

        Args:
            filter_ancillas: If set to ``True`` only qubits in the input circuit
                will be in the returned layout. Any ancilla qubits added to the
                output circuit will be filtered from the returned object.
        Returns:
            A layout object mapping the input circuit's :class:`~.circuit.Qubit`
            objects to the positions of the selected physical qubits.
        """
        if not filter_ancillas:
            return self.initial_layout
        return Layout(
            {
                k: v
                for k, v in self.initial_layout.get_virtual_bits().items()
                if self.input_qubit_mapping[k] < self._input_qubit_count
            }
        )

    def initial_index_layout(self, filter_ancillas: bool = False) -> list[int]:
        """Generate an initial layout as an array of integers.

        Args:
            filter_ancillas: If set to ``True`` any ancilla qubits added
                to the transpiler will not be included in the output.

        Return:
            A layout array that maps a position in the array to its new position in the output
            circuit.
        """

        virtual_map = self.initial_layout.get_virtual_bits()
        if filter_ancillas:
            output = [None] * self._input_qubit_count
        else:
            output = [None] * len(virtual_map)
        for virt, phys in virtual_map.items():
            pos = self.input_qubit_mapping[virt]
            if filter_ancillas and pos >= self._input_qubit_count:
                continue
            output[pos] = phys
        return output

    def routing_permutation(self) -> list[int]:
        """Generate a final layout as an array of integers.

        If there is no :attr:`.final_layout` attribute present then that indicates
        there was no output permutation caused by routing or other transpiler
        transforms. In this case the function will return a list of ``[0, 1, 2, .., n]``.

        Returns:
            A layout array that maps a position in the array to its new position in the output
            circuit.
        """
        if self.final_layout is None:
            return list(range(len(self._output_qubit_list)))
        virtual_map = self.final_layout.get_virtual_bits()
        return [virtual_map[virt] for virt in self._output_qubit_list]

    def final_index_layout(self, filter_ancillas: bool = True) -> list[int]:
        """Generate the final layout as an array of integers.

        This method will generate an array of final positions for each qubit in the input circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            tqc.h(2)
            tqc.cx(2, 1)
            tqc.swap(0, 1)
            tqc.cx(2, 1)

        then the :meth:`.final_index_layout` method returns::

            [2, 0, 1]

        This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
        in the output circuit during the layout stage, which is mapped to qubit 2 during the
        routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
        circuit during the layout stage, which is mapped to qubit 0 during the routing
        stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
        during the layout stage, which is mapped to qubit 1 during the routing stage.
        The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A list of final positions for each input circuit qubit.
        """
        if self._input_qubit_count is None:
            # TODO: After there is a way to differentiate the ancilla qubits added by the transpiler
            # don't use the ancilla name anymore.See #10817 for discussion on this.
            num_source_qubits = len(
                [
                    x
                    for x in self.input_qubit_mapping
                    if getattr(x, "_register", "").startswith("ancilla")
                ]
            )
        else:
            num_source_qubits = self._input_qubit_count
        if self._output_qubit_list is None:
            circuit_qubits = list(self.final_layout.get_virtual_bits())
        else:
            circuit_qubits = self._output_qubit_list

        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        qubit_indices = []
        if filter_ancillas:
            num_qubits = num_source_qubits
        else:
            num_qubits = len(self._output_qubit_list)
        for index in range(num_qubits):
            qubit_idx = self.initial_layout[pos_to_virt[index]]
            if self.final_layout is not None:
                qubit_idx = self.final_layout[circuit_qubits[qubit_idx]]
            qubit_indices.append(qubit_idx)
        return qubit_indices

    def final_virtual_layout(self, filter_ancillas: bool = True) -> Layout:
        """Generate the final layout as a :class:`.Layout` object.

        This method will generate an array of final positions for each qubit in the input circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            tqc.h(2)
            tqc.cx(2, 1)
            tqc.swap(0, 1)
            tqc.cx(2, 1)

        then the return from this function would be a layout object::

            Layout({
                qc.qubits[0]: 2,
                qc.qubits[1]: 0,
                qc.qubits[2]: 1,
            })

        This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
        in the output circuit during the layout stage, which is mapped to qubit 2 during the
        routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
        circuit during the layout stage, which is mapped to qubit 0 during the routing
        stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
        during the layout stage, which is mapped to qubit 1 during the routing stage.
        The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A layout object mapping to the final positions for each qubit.
        """
        res = self.final_index_layout(filter_ancillas=filter_ancillas)
        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        return Layout({pos_to_virt[index]: phys for index, phys in enumerate(res)})

    @classmethod
    def from_property_set(
        cls, dag: DAGCircuit, property_set: PropertySet
    ) -> TranspileLayout | None:
        """Construct the :class:`TranspileLayout` by reading out the fields from the given
        :class:`.PropertySet`.  Returns ``None`` if there are no layout-setting keys present.

        This includes combining the different keys of the property set into the full set of initial
        and final layouts, including virtual permutations.

        This does not invalidate or in any way mutate the given property set.  In order to
        "canonicalize" the property set afterwards, call :meth:`write_into_property_set`.

        This reads the following property-set keys:

        ``layout``
            **Required**. The :class:`.Layout` object mapping virtual qubits (potentially expanded
            with ancillas) to physical-qubit indices.  This corresponds directly to
            :attr:`initial_layout`.

            .. note::
                In all standard use, this is a required field.  However, if
                ``virtual_permutation_layout`` is set, then a "trivial" layout will be inferred,
                even if the circuit is not actually laid out to hardware.  This is an unfortunate
                limitation of this class's data model, where it is not possible to specify a final
                permutation without also having an initial layout. This deficiency will be corrected
                in Qiskit 3.0.

        ``original_qubit_indices``
            **Required** (but automatically set by the :class:`.PassManager`).  The mapping
            ``{virtual: index}`` that indicates which relative index each incoming virtual qubit
            was, in the input circuit.  This can be expanded with ancillas too (in which case the
            ancilla indices don't mean much, since they weren't in the incoming circuit).

        ``num_input_qubits``
            **Required** (but automatically set by the :class:`.PassManager`).  The number of
            explicit virtual qubits in the input circuit (i.e. not including implicit ancillas).

        ``final_layout``
            **Optional**.  The effective final permutation, in terms of the current qubits of the
            :class:`.DAGCircuit`.  This corresponds directly to :attr:`final_layout`.

        ``virtual_permutation_layout``
            **Optional**.  This is set by certain optimization passes that run before layout
            selection, such as :class:`.ElidePermutations`.  It is similar in spirit to
            ``final_layout``, but typically only applies to the input virtual qubits.

            .. warning::
                This object uses the opposite permutation convention to ``final_layout`` due to an
                oversight in Qiskit during its introduction.  In other words,
                ``virtual_permutation_layout`` maps a :class:`.Qubit` instance at the end of the
                circuit to its integer index at the start of the circuit.

        Args:
            dag: the current state of the :class:`.DAGCircuit`.
            property_set: the current transpiler's property set.  This must at least have the
                ``layout`` key set.
        """
        initial_layout = property_set["layout"]
        final_layout = property_set["final_layout"]
        input_qubit_indices = property_set["original_qubit_indices"]
        virtual_permutation_layout = property_set["virtual_permutation_layout"]
        num_input_qubits = property_set["num_input_qubits"]

        output_qubits = list(dag.qubits)

        if initial_layout is None and virtual_permutation_layout is None and final_layout is None:
            # Nothing that truly sets a Python-space `TranspileLayout` is set.
            return None
        if initial_layout is not None and virtual_permutation_layout is None:
            # This is the "happy" path where everything is already (in theory) normalised to the
            # original state of how the transpiler handled these properties.
            return cls(
                initial_layout, input_qubit_indices, final_layout, num_input_qubits, output_qubits
            )

        # Due to current (at least as of Qiskit 2.x) limitations of `TranspileLayout`, the only
        # way to return a routing permutation if `virtual_permutation_layout` is set is to force
        # an initial layout, even if there isn't actually any laying out to hardware.
        if initial_layout is None:
            initial_layout = Layout(dict(enumerate(dag.qubits)))
        if virtual_permutation_layout is None:
            virtual_permutation_layout = Layout(input_qubit_indices)
        if final_layout is None:
            final_layout = Layout(dict(enumerate(dag.qubits)))
        input_qubits = sorted(input_qubit_indices, key=input_qubit_indices.get)

        num_qubits = len(dag.qubits)

        # Throughout the rest of this, we will speak about index permutations as lists that mean:
        #
        #    qubit `permutation[i]` goes to new index `i`
        #
        # or in alternative language,
        #
        #   after the permutation, qubit `i` contains qubit `permutation[i]`.
        #
        # This is to match the convention that `PermutationGate` uses, but beware: it might not be
        # the way you think about permutations (it's not my preferred convention---Jake).
        #
        # Now, we'll step through the transpilation process.  At each point, we'll relate the
        # objects we have back to a 3-tuple of abstract objects, which are applied in order:
        #
        #   (relabelling, explicit instructions, implicit instructions)
        #
        # The "explicit instructions" are always just the DAG itself.  The "relabelling" is
        # generally associated with the "initial layout" and the metadata linking the original
        # virtual qubit objects and their indices.  The "implicit instructions" is where all the
        # interesting stuff happens; at the moment, in Qiskit, we only track an implicit final
        # permutation, though you could imagine a world where we allow a lot more things to be
        # tracked, such as necessary classical post-processing steps.
        #
        # We will attempt to always have in hand the permutation that needs to be appended to the
        # current explicit circuit to "undo" all the elided/added permutations.  For example, we
        # want the permutation that adds back in what `ElidePermutations` might have removed, or
        # "undoes" the swaps that routing added.  Explicitly, we want to have a ``permutation`` such
        # that this sequence of operations brings us back to the same semantics as the original
        # virtual circuit:
        #
        #   current = <current explicit circuit/DAG>
        #   # Make the permutation explicit; the permutation is defined on the current qubit labels.
        #   current.append(PermutationGate(permutation), current.qubits)
        #   # Now revert the `initial_layout` relabelling.
        #   relabel_qubits_from_physical_to_virtual(qc, initial_layout)
        #
        # where "semantics" would mean exact unitary equivalence for a unitary input, and something
        # a bit hand-wavier once measurements are involved.

        # First, virtual permutation modifications happen.  For example, `ElidePermutations` or
        # `StarPreRouting`.  Note that `virtual_permutation_layout` uses an opposite convention to
        # `final_layout` for defining the permutation.
        undo_elided_on_virtuals = [
            virtual_permutation_layout[virtual_bit]
            for virtual_bit in input_qubits[:num_input_qubits]
        ]
        # `virtual_permutation_layout` is defined without ancillas.  If they got added later, extend
        # the virtual permutation with the implicit identity on the other components.
        if num_qubits > len(undo_elided_on_virtuals):
            undo_elided_on_virtuals.extend(range(len(undo_elided_on_virtuals), num_qubits))

        def relabel_virtual_to_physical(virtual_index: int):
            return initial_layout[input_qubits[virtual_index]]

        def relabel_physical_to_virtual(physical_index: int):
            return input_qubit_indices[initial_layout[physical_index]]

        # Next, a layout pass runs, and maps the virtual qubits to physical qubits.  We want to
        # update our permutation so that it can be applied to the _physical_ circuit instead.  This
        # means relabelling both references to circuit indices: the actual values in the list, but
        # also the indices in the list that they're located at.
        undo_elided_on_physicals = [
            relabel_virtual_to_physical(
                undo_elided_on_virtuals[relabel_physical_to_virtual(physical_index)]
            )
            for physical_index in range(num_qubits)
        ]

        # Next, routing runs.  This adds in an extra permutation, which comes between "the circuit"
        # and the "undoing permutation" we just calculated.  Routing returns the total
        # permutation that it has applied, so if we send the "index before routing" to "index after
        # routing", our permutation will then properly undo everything.  Note that `final_layout`
        # uses the opposite permutation convention to `virtual_permutation_layout`.
        undo_routing_on_physicals = [
            dag.find_bit(final_layout[physical_index]).index for physical_index in range(num_qubits)
        ]
        undo_total_on_physicals = [
            undo_elided_on_physicals[undo_routing_on_physicals[physical_index]]
            for physical_index in range(num_qubits)
        ]

        # Finally, turn what we have into the same convention that `final_layout` uses.
        final_layout = Layout(
            {
                qubit_index: dag.qubits[is_set_to]
                for qubit_index, is_set_to in enumerate(undo_total_on_physicals)
            }
        )

        return cls(
            initial_layout, input_qubit_indices, final_layout, num_input_qubits, list(dag.qubits)
        )

    def write_into_property_set(self, property_set: dict[str, object]):
        """'Unpack' this layout into the loose-constraints form of the ``property_set``.

        This is the inverse method of :meth:`from_property_set`.

        This always writes the follow property-set keys, overwriting them if they were already set:

        ``layout``
            Directly corresponds to :attr:`initial_layout`.

        ``original_qubit_indices``
            Directly corresponds to :attr:`input_qubit_mapping`.

        ``final_layout``
            Directly corresponds to :attr:`final_layout`.  Note that this might not be identical to
            the ``final_layout`` from before a call to :meth:`from_property_set`, because the
            effects of ``virtual_permutation_layout`` will have been combined into it.

        ``virtual_permutation_layout``
            Deleted from the property set; :class:`TranspileLayout` "finalizes" the multiple
            separate permutations into one single permutation, to retain the canonical form.

        In addition, the following keys are updated, if this :class:`TranspileLayout` has a known
        value for them.  They are left as-is if not, to handle cases where this class was manually
        constructed without setting certain optional fields.

        ``num_input_qubits``
            The number of non-ancilla virtual qubits in the input circuit.

        Args:
            property_set: the :class:`.PropertySet` (or general :class:`dict`) that the output
                should be written into.  This mutates the input in place.
        """
        for always_overwrite in (
            "layout",
            "final_layout",
            "original_qubit_indices",
            "virtual_permutation_layout",
        ):
            property_set.pop(always_overwrite, None)

        property_set["layout"] = self.initial_layout.copy()
        property_set["original_qubit_indices"] = self.input_qubit_mapping.copy()
        if self.final_layout is not None:
            property_set["final_layout"] = self.final_layout.copy()
        if self._input_qubit_count is not None:
            property_set["num_input_qubits"] = self._input_qubit_count
