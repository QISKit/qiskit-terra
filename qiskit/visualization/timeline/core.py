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

"""
Core module of the timeline drawer.

This module provides the `DrawerCanvas` which is a collection of drawings.
The canvas instance is not just a container of drawing objects, as it also performs
data processing like binding abstract coordinates.


Initialization
~~~~~~~~~~~~~~
The `DataCanvas` is not exposed to users as they are implicitly initialized in the
interface function. It is noteworthy that the data canvas is agnostic to plotters.
This means once the canvas instance is initialized we can reuse this data
among multiple plotters. The canvas is initialized with a stylesheet.

    ```python
    canvas = DrawerCanvas(stylesheet=stylesheet)
    canvas.load_program(sched)
    canvas.update()
    ```

Once all properties are set, `.update` method is called to apply changes to drawings.

Update
~~~~~~
To update the image, a user can set new values to canvas and then call the `.update` method.

    ```python
    canvas.set_time_range(2000, 3000)
    canvas.update()
    ```

All stored drawings are updated accordingly. The plotter API can access to
drawings with `.collections` property of the canvas instance. This returns
an iterator of drawings with the unique data key.
If a plotter provides object handler for plotted shapes, the plotter API can manage
the lookup table of the handler and the drawings by using this data key.
"""
from __future__ import annotations
import warnings
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from enum import Enum

import numpy as np

from qiskit import circuit
from qiskit.transpiler.target import Target
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawings, types
from qiskit.visualization.timeline.stylesheet import QiskitTimelineStyle


class DrawerCanvas:
    """Data container for drawings."""

    def __init__(self, stylesheet: QiskitTimelineStyle):
        """Create new data container."""
        # stylesheet
        self.formatter = stylesheet.formatter
        self.generator = stylesheet.generator
        self.layout = stylesheet.layout

        # drawings
        self._collections: dict[str, drawings.ElementaryData] = {}
        self._output_dataset: dict[str, drawings.ElementaryData] = {}

        # vertical offset of bits
        self.bits: list[types.Bits] = []
        self.assigned_coordinates: dict[types.Bits, float] = {}

        # visible controls
        self.disable_bits: set[types.Bits] = set()
        self.disable_types: set[str] = set()

        # time
        self._time_range = (0, 0)

        # graph height
        self.vmax = 0
        self.vmin = 0

    @property
    def time_range(self) -> tuple[int, int]:
        """Return current time range to draw.

        Calculate net duration and add side margin to edge location.

        Returns:
            Time window considering side margin.
        """
        t0, t1 = self._time_range

        duration = t1 - t0
        new_t0 = t0 - duration * self.formatter["margin.left_percent"]
        new_t1 = t1 + duration * self.formatter["margin.right_percent"]

        return new_t0, new_t1

    @time_range.setter
    def time_range(self, new_range: tuple[int, int]):
        """Update time range to draw."""
        self._time_range = new_range

    @property
    def collections(self) -> Iterator[tuple[str, drawings.ElementaryData]]:
        """Return currently active entries from drawing data collection.

        The object is returned with unique name as a key of an object handler.
        When the horizontal coordinate contains `AbstractCoordinate`,
        the value is substituted by current time range preference.
        """
        yield from self._output_dataset.items()

    def add_data(self, data: drawings.ElementaryData):
        """Add drawing to collections.

        If the given object already exists in the collections,
        this interface replaces the old object instead of adding new entry.

        Args:
            data: New drawing to add.
        """
        if not self.formatter["control.show_clbits"]:
            data.bits = [b for b in data.bits if not isinstance(b, circuit.Clbit)]
        self._collections[data.data_key] = data

    def load_program(self, program: circuit.QuantumCircuit, target: Target | None = None):
        """Load quantum circuit and create drawing..

        .. deprecated:: 1.3
           Visualization of unscheduled circuits with the timeline drawer has been
           deprecated in Qiskit 1.3.
           This circuit should be transpiled with a scheduler, despite having instructions
           with explicit durations.

        .. deprecated:: 1.3
           Targets with duration-less operations are going to error in Qiskit 2.0.

        Args:
            program: Scheduled circuit object to draw.
            target: The target the circuit is scheduled for. This contains backend information
                including the instruction durations used in scheduling.

        Raises:
           VisualizationError: When circuit is not scheduled.
        """
        not_gate_like = (circuit.Barrier,)

        if getattr(program, "_op_start_times") is None:
            # Run scheduling for backward compatibility
            from qiskit import transpile
            from qiskit.transpiler import TranspilerError

            warnings.warn(
                "Visualizing un-scheduled circuit with timeline drawer has been deprecated. "
                "This circuit should be transpiled with scheduler though it consists of "
                "instructions with explicit durations.",
                DeprecationWarning,
                stacklevel=3,
            )

            try:
                program = transpile(
                    program,
                    scheduling_method="alap",
                    target=target,
                    optimization_level=0,
                )
            except TranspilerError as ex:
                raise VisualizationError(
                    f"Input circuit {program.name} is not scheduled and it contains "
                    "operations with unknown delays. This cannot be visualized."
                ) from ex

        for t0, instruction in zip(program.op_start_times, program.data):
            bits = list(instruction.qubits) + list(instruction.clbits)
            for bit_pos, bit in enumerate(bits):
                if not isinstance(instruction.operation, not_gate_like):
                    # Generate draw object for gates
                    if target is not None:
                        duration = None
                        op_props = target.get(instruction.operation.name)
                        if op_props is not None:
                            inst_props = op_props.get(
                                tuple(program.find_bit(x).index for x in instruction.qubits)
                            )
                            if inst_props is not None:
                                duration = getattr(inst_props, "duration")
                                if duration is not None:
                                    duration = target.seconds_to_dt(duration)

                        if instruction.name == "delay":
                            duration = instruction.operation.duration
                        if duration is None:
                            # Warn here because an incomplete target isn't obvious most of the time
                            raise VisualizationError(
                                "Target doesn't contain a duration for "
                                f"{instruction.operation.name} on {bit_pos}."
                            )
                    elif instruction.name == "delay":
                        duration = instruction.operation.duration
                    else:
                        raise VisualizationError(
                            "Target not specified this is required to provide instruction timing"
                        )
                    gate_source = types.ScheduledGate(
                        t0=t0,
                        operand=instruction.operation,
                        duration=duration,
                        bits=bits,
                        bit_position=bit_pos,
                    )
                    for gen in self.generator["gates"]:
                        if getattr(gen, "accepts_program", False):
                            gen = partial(gen, program=program)
                        for datum in gen(gate_source, formatter=self.formatter):
                            self.add_data(datum)
                    if len(bits) > 1 and bit_pos == 0:
                        # Generate draw object for gate-gate link
                        line_pos = t0 + 0.5 * duration
                        link_source = types.GateLink(
                            t0=line_pos, opname=instruction.operation.name, bits=bits
                        )
                        for gen in self.generator["gate_links"]:
                            if getattr(gen, "accepts_program", False):
                                gen = partial(gen, program=program)
                            for datum in gen(link_source, formatter=self.formatter):
                                self.add_data(datum)
                if isinstance(instruction.operation, circuit.Barrier):
                    # Generate draw object for barrier
                    barrier_source = types.Barrier(t0=t0, bits=bits, bit_position=bit_pos)
                    for gen in self.generator["barriers"]:
                        if getattr(gen, "accepts_program", False):
                            gen = partial(gen, program=program)
                        for datum in gen(barrier_source, formatter=self.formatter):
                            self.add_data(datum)

        self.bits = list(program.qubits) + list(program.clbits)
        for bit in self.bits:
            for gen in self.generator["bits"]:
                # Generate draw objects for bit
                if getattr(gen, "accepts_program", False):
                    gen = partial(gen, program=program)
                for datum in gen(bit, formatter=self.formatter):
                    self.add_data(datum)

        # update time range
        t_end = max(program._duration, self.formatter["margin.minimum_duration"])
        self.set_time_range(t_start=0, t_end=t_end)

    def set_time_range(self, t_start: int, t_end: int):
        """Set time range to draw.

        Args:
            t_start: Left boundary of drawing in units of cycle time.
            t_end: Right boundary of drawing in units of cycle time.
        """
        self.time_range = (t_start, t_end)

    def set_disable_bits(self, bit: types.Bits, remove: bool = True):
        """Interface method to control visibility of bits.

        Specified object in the blocked list will not be shown.

        Args:
            bit: A qubit or classical bit object to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if remove:
            self.disable_bits.add(bit)
        else:
            self.disable_bits.discard(bit)

    def set_disable_type(self, data_type: types.DataTypes, remove: bool = True):
        """Interface method to control visibility of data types.

        Specified object in the blocked list will not be shown.

        Args:
            data_type: A drawing data type to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if isinstance(data_type, Enum):
            data_type_str = str(data_type.value)
        else:
            data_type_str = data_type

        if remove:
            self.disable_types.add(data_type_str)
        else:
            self.disable_types.discard(data_type_str)

    def update(self):
        """Update all collections.

        This method should be called before the canvas is passed to the plotter.
        """
        self._output_dataset.clear()
        self.assigned_coordinates.clear()

        # update coordinate
        y0 = -self.formatter["margin.top"]
        for bit in self.layout["bit_arrange"](self.bits):
            # remove classical bit
            if isinstance(bit, circuit.Clbit) and not self.formatter["control.show_clbits"]:
                continue
            # remove idle bit
            if not self._check_bit_visible(bit):
                continue
            offset = y0 - 0.5
            self.assigned_coordinates[bit] = offset
            y0 = offset - 0.5
        self.vmax = 0
        self.vmin = y0 - self.formatter["margin.bottom"]

        # add data
        temp_gate_links = {}
        temp_data = {}
        for data_key, data in self._collections.items():
            # deep copy to keep original data hash
            new_data = deepcopy(data)
            new_data.xvals = self._bind_coordinate(data.xvals)
            new_data.yvals = self._bind_coordinate(data.yvals)
            if data.data_type == str(types.LineType.GATE_LINK.value):
                temp_gate_links[data_key] = new_data
            else:
                temp_data[data_key] = new_data

        # update horizontal offset of gate links
        temp_data.update(self._check_link_overlap(temp_gate_links))

        # push valid data
        for data_key, data in temp_data.items():
            if self._check_data_visible(data):
                self._output_dataset[data_key] = data

    def _check_data_visible(self, data: drawings.ElementaryData) -> bool:
        """A helper function to check if the data is visible.

        Args:
            data: Drawing object to test.

        Returns:
            Return `True` if the data is visible.
        """
        _barriers = [str(types.LineType.BARRIER.value)]

        _delays = [str(types.BoxType.DELAY.value), str(types.LabelType.DELAY.value)]

        def _time_range_check(_data):
            """If data is located outside the current time range."""
            t0, t1 = self.time_range
            if np.max(_data.xvals) < t0 or np.min(_data.xvals) > t1:
                return False
            return True

        def _associated_bit_check(_data):
            """If any associated bit is not shown."""
            if all(bit not in self.assigned_coordinates for bit in _data.bits):
                return False
            return True

        def _data_check(_data):
            """If data is valid."""
            if _data.data_type == str(types.LineType.GATE_LINK.value):
                active_bits = [bit for bit in _data.bits if bit not in self.disable_bits]
                if len(active_bits) < 2:
                    return False
            elif _data.data_type in _barriers and not self.formatter["control.show_barriers"]:
                return False
            elif _data.data_type in _delays and not self.formatter["control.show_delays"]:
                return False
            return True

        checks = [_time_range_check, _associated_bit_check, _data_check]
        if all(check(data) for check in checks):
            return True

        return False

    def _check_bit_visible(self, bit: types.Bits) -> bool:
        """A helper function to check if the bit is visible.

        Args:
            bit: Bit object to test.

        Returns:
            Return `True` if the bit is visible.
        """
        _gates = [str(types.BoxType.SCHED_GATE.value), str(types.SymbolType.FRAME.value)]

        if bit in self.disable_bits:
            return False

        if self.formatter["control.show_idle"]:
            return True

        for data in self._collections.values():
            if bit in data.bits and data.data_type in _gates:
                return True
        return False

    def _bind_coordinate(self, vals: Iterator[types.Coordinate]) -> np.ndarray:
        """A helper function to bind actual coordinates to an `AbstractCoordinate`.

        Args:
            vals: Sequence of coordinate objects associated with a drawing.

        Returns:
            Numpy data array with substituted values.
        """

        def substitute(val: types.Coordinate):
            if val == types.AbstractCoordinate.LEFT:
                return self.time_range[0]
            if val == types.AbstractCoordinate.RIGHT:
                return self.time_range[1]
            if val == types.AbstractCoordinate.TOP:
                return self.vmax
            if val == types.AbstractCoordinate.BOTTOM:
                return self.vmin
            raise VisualizationError(f"Coordinate {val} is not supported.")

        try:
            return np.asarray(vals, dtype=float)
        except TypeError:
            return np.asarray(list(map(substitute, vals)), dtype=float)

    def _check_link_overlap(
        self, links: dict[str, drawings.GateLinkData]
    ) -> dict[str, drawings.GateLinkData]:
        """Helper method to check overlap of bit links.

        This method dynamically shifts horizontal position of links if they are overlapped.
        """
        duration = self.time_range[1] - self.time_range[0]
        allowed_overlap = self.formatter["margin.link_interval_percent"] * duration

        # return y coordinates
        def y_coords(link: drawings.GateLinkData):
            return np.array([self.assigned_coordinates.get(bit, np.nan) for bit in link.bits])

        # group overlapped links
        overlapped_group: list[list[str]] = []
        data_keys = list(links.keys())
        while len(data_keys) > 0:
            ref_key = data_keys.pop()
            overlaps = set()
            overlaps.add(ref_key)
            for key in data_keys[::-1]:
                # check horizontal overlap
                if np.abs(links[ref_key].xvals[0] - links[key].xvals[0]) < allowed_overlap:
                    # check vertical overlap
                    y0s = y_coords(links[ref_key])
                    y1s = y_coords(links[key])
                    v1 = np.nanmin(y0s) - np.nanmin(y1s)
                    v2 = np.nanmax(y0s) - np.nanmax(y1s)
                    v3 = np.nanmin(y0s) - np.nanmax(y1s)
                    v4 = np.nanmax(y0s) - np.nanmin(y1s)
                    if not (v1 * v2 > 0 and v3 * v4 > 0):
                        overlaps.add(data_keys.pop(data_keys.index(key)))
            overlapped_group.append(list(overlaps))

        # renew horizontal offset
        new_links = {}
        for overlaps in overlapped_group:
            if len(overlaps) > 1:
                xpos_mean = np.mean([links[key].xvals[0] for key in overlaps])
                # sort link key by y position
                sorted_keys = sorted(overlaps, key=lambda x: np.nanmax(y_coords(links[x])))
                x0 = xpos_mean - 0.5 * allowed_overlap * (len(overlaps) - 1)
                for ind, key in enumerate(sorted_keys):
                    data = links[key]
                    data.xvals = [x0 + ind * allowed_overlap]
                    new_links[key] = data
            else:
                key = overlaps[0]
                new_links[key] = links[key]

        return {key: new_links[key] for key in links.keys()}
