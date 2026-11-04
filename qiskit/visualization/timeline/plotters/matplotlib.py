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


"""Matplotlib plotter API."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import core, types, drawings
from qiskit.visualization.timeline.plotters.base_plotter import BasePlotter
from qiskit.visualization.utils import matplotlib_close_if_inline


class MplPlotter(BasePlotter):
    """Matplotlib API for pulse drawer.

    This plotter arranges bits along y axis of 2D canvas with vertical offset.
    """

    def __init__(self, canvas: core.DrawerCanvas, axis: plt.Axes | None = None):
        """Create new plotter.

        Args:
            canvas: Configured drawer canvas object. Canvas object should be updated
                with `.update` method before initializing the plotter.
            axis: Matplotlib axis object. When `axis` is provided, the plotter updates
                given axis instead of creating and returning new matplotlib figure.
        """
        super().__init__(canvas=canvas)

        if axis is None:
            fig_height = self.canvas.vmax - self.canvas.vmin
            fig_h = self.canvas.formatter["general.fig_unit_height"] * fig_height
            fig_w = self.canvas.formatter["general.fig_width"]

            self.figure = plt.figure(figsize=(fig_w, fig_h))
            self.ax = self.figure.add_subplot(1, 1, 1)
        else:
            self.figure = axis.figure
            self.ax = axis

        self.initialize_canvas()

    def initialize_canvas(self):
        """Format appearance of matplotlib canvas."""
        self.ax.set_facecolor(self.canvas.formatter["color.background"])

        # axis lines
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.spines["top"].set_color("none")

        # axis labels
        self.ax.set_yticks([])
        axis_config = self.canvas.layout["time_axis_map"](time_window=self.canvas.time_range)

        self.ax.set_xticks(list(axis_config.axis_map.keys()))
        self.ax.set_xticklabels(
            list(axis_config.axis_map.values()),
            fontsize=self.canvas.formatter["text_size.axis_label"],
        )
        self.ax.set_xlabel(
            axis_config.label, fontsize=self.canvas.formatter["text_size.axis_label"]
        )

        # boundary
        self.ax.set_xlim(*self.canvas.time_range)
        self.ax.set_ylim(self.canvas.vmin, self.canvas.vmax)

    def draw(self):
        """Output drawings stored in canvas object."""

        for _, data in self.canvas.collections:
            xvals = np.asarray(data.xvals, dtype=float)
            yvals = np.asarray(data.yvals, dtype=float)
            offsets = [self.canvas.assigned_coordinates[bit] for bit in data.bits]

            if isinstance(data, drawings.BoxData):
                # box data
                if data.data_type in [
                    str(types.BoxType.SCHED_GATE.value),
                    str(types.BoxType.DELAY.value),
                ]:
                    # draw a smoothly rounded rectangle
                    xs, ys1, ys2 = self._time_bucket_outline(xvals, yvals)
                    self.ax.fill_between(
                        x=xs, y1=ys1 + offsets[0], y2=ys2 + offsets[0], **data.styles
                    )

                else:
                    # draw a rectangle
                    x0, x1 = xvals
                    y0, y1 = yvals + offsets[0]

                    rect = Rectangle(xy=(x0, y0), width=x1 - x0, height=y1 - y0)
                    pc = PatchCollection([rect], **data.styles)
                    self.ax.add_collection(pc)

            elif isinstance(data, drawings.LineData):
                # line data
                self.ax.plot(xvals, yvals + offsets[0], **data.styles)

            elif isinstance(data, drawings.TextData):
                # text data
                if data.latex is not None:
                    s = rf"${data.latex}$"
                else:
                    s = data.text

                self.ax.text(x=xvals[0], y=yvals[0] + offsets[0], s=s, **data.styles)

            elif isinstance(data, drawings.GateLinkData):
                # gate link data
                self.ax.plot(xvals.repeat(len(offsets)), offsets, **data.styles)

            else:
                raise VisualizationError(
                    f"Data {data} is not supported by {self.__class__.__name__}"
                )

    def _time_bucket_outline(
        self, xvals: np.ndarray, yvals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate outline of time bucket. Edges are smoothly faded.

        Args:
            xvals: Left and right point coordinates.
            yvals: Bottom and top point coordinates.

        Returns:
            Coordinate vectors of time bucket fringe.
        """
        x0, x1 = xvals
        y0, y1 = yvals

        width = x1 - x0
        y_mid = 0.5 * (y0 + y1)

        risefall = int(min(self.canvas.formatter["time_bucket.edge_dt"], max(width / 2 - 2, 0)))
        edge = np.sin(np.pi / 2 * np.arange(0, risefall) / risefall)

        xs = np.concatenate(
            [
                np.arange(x0, x0 + risefall),
                [x0 + risefall, x1 - risefall],
                np.arange(x1 - risefall + 1, x1 + 1),
            ]
        )

        l1 = (y1 - y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])
        l2 = (y0 - y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])

        return xs, l1, l2

    def save_file(self, filename: str):
        """Save image to file.
        Args:
            filename: File path to output image data.
        """
        plt.savefig(filename, bbox_inches="tight", dpi=self.canvas.formatter["general.dpi"])

    def get_image(self, interactive: bool = False) -> matplotlib.pyplot.Figure:
        """Get image data to return.
        Args:
            interactive: When set `True` show the circuit in a new window.
                This depends on the matplotlib backend being used supporting this.
        Returns:
            Matplotlib figure data.
        """
        matplotlib_close_if_inline(self.figure)

        if self.figure and interactive:
            self.figure.show()
        try:
            self.figure.tight_layout()
        except AttributeError:
            pass
        return self.figure
