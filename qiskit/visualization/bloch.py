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


# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""Bloch sphere"""

__all__ = ["Bloch"]

import math
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D

from .utils import matplotlib_close_if_inline


# This version pattern is taken from the pypa packaging project:
# https://github.com/pypa/packaging/blob/21.3/packaging/version.py#L223-L254
# which is dual licensed Apache 2.0 and BSD see the source for the original
# authors and other details
VERSION_PATTERN = (
    "^"
    + r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""
    + "$"
)
VERSION_PATTERN_REGEX = re.compile(VERSION_PATTERN, re.VERBOSE | re.IGNORECASE)


class Arrow3D(Patch3D, FancyArrowPatch):
    """Makes a fancy arrow"""

    # Nasty hack around a poorly implemented deprecation warning in Matplotlib 3.5 that issues two
    # deprecation warnings if an artist's module does not claim to be part of the below module.
    # This revolves around the method `Patch3D.do_3d_projection(self, renderer=None)`.  The
    # `renderer` argument has been deprecated since Matplotlib 3.4, but in 3.5 some internal calls
    # during `Axes3D` display started calling the method.  If an artist does not have this module,
    # then it issues a deprecation warning, and calls it by passing the `renderer` parameter as
    # well, which consequently triggers another deprecation warning.  We should be able to remove
    # this once 3.6 is the minimum supported version, because the deprecation period ends then.
    __module__ = "mpl_toolkits.mplot3d.art3d"

    def __init__(self, xs, ys, zs, zdir="z", **kwargs):
        # The Patch3D.__init__() method just calls its own super() method and then
        # self.set_3d_properties, but its __init__ signature is actually pretty incompatible with
        # how it goes on to call set_3d_properties, so we just have to do things ourselves.  The
        # parent of Patch3D is Patch, which is also a parent of FancyArrowPatch, so its __init__ is
        # still getting suitably called.

        FancyArrowPatch.__init__(self, (0, 0), (0, 0), **kwargs)
        self.set_3d_properties(tuple(zip(xs, ys)), zs, zdir)
        self._path2d = None

    def draw(self, renderer):
        xs3d, ys3d, zs3d = zip(*self._segment3d)
        x_s, y_s, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self._path2d = matplotlib.path.Path(np.column_stack([x_s, y_s]))
        self.set_positions((x_s[0], y_s[0]), (x_s[1], y_s[1]))
        FancyArrowPatch.draw(self, renderer)


class Bloch:
    """Class for plotting data on the Bloch sphere.  Valid data can be
    either points or vectors.

    Attributes:
        axes (instance):
            User supplied Matplotlib axes for Bloch sphere animation.
        fig (instance):
            User supplied Matplotlib Figure instance for plotting Bloch sphere.
        font_color (str):
            Color of font used for Bloch sphere labels.
        font_size (int):
            Size of font used for Bloch sphere labels.
        frame_alpha (float):
            Sets transparency of Bloch sphere frame.
        frame_color (str):
            Color of sphere wireframe.
        frame_width (int):
            Width of wireframe.
        point_color (list):
            List of colors for Bloch sphere point markers to cycle through.
            i.e. By default, points 0 and 4 will both be blue ('b').
        point_marker (list):
            List of point marker shapes to cycle through.
        point_size (list):
            List of point marker sizes. Note, not all point markers look
            the same size when plotted!
        sphere_alpha (float):
            Transparency of Bloch sphere itself.
        sphere_color (str):
            Color of Bloch sphere.
        figsize (list):
            Figure size of Bloch sphere plot.  Best to have both numbers the same;
            otherwise you will have a Bloch sphere that looks like a football.
        vector_color (list):
            List of vector colors to cycle through.
        vector_width (int):
            Width of displayed vectors.
        vector_style (str):
            Vector arrowhead style (from matplotlib's arrow style).
        vector_mutation (int):
            Width of vectors arrowhead.
        view (list):
            Azimuthal and Elevation viewing angles.
        xlabel (list):
            List of strings corresponding to +x and -x axes labels, respectively.
        xlpos (list):
            Positions of +x and -x labels respectively.
        ylabel (list):
            List of strings corresponding to +y and -y axes labels, respectively.
        ylpos (list):
            Positions of +y and -y labels respectively.
        zlabel (list):
            List of strings corresponding to +z and -z axes labels, respectively.
        zlpos (list):
            Positions of +z and -z labels respectively.
    """

    def __init__(
        self, fig=None, axes=None, view=None, figsize=None, background=False, font_size=20
    ):

        # Figure and axes
        self._ext_fig = False
        if fig is not None:
            self._ext_fig = True
        self.fig = fig
        self._ext_axes = False
        if axes is not None:
            self._ext_fig = True
            self._ext_axes = True
        self.axes = axes
        # Background axes, default = False
        self.background = background
        # The size of the figure in inches, default = [5,5].
        self.figsize = figsize if figsize else [5, 5]
        # Azimuthal and Elevation viewing angles, default = [-60,30].
        self.view = view if view else [-60, 30]
        # Color of Bloch sphere, default = #FFDDDD
        self.sphere_color = "#FFDDDD"
        # Transparency of Bloch sphere, default = 0.2
        self.sphere_alpha = 0.2
        # Color of wireframe, default = 'gray'
        self.frame_color = "gray"
        # Width of wireframe, default = 1
        self.frame_width = 1
        # Transparency of wireframe, default = 0.2
        self.frame_alpha = 0.2
        # Labels for x-axis (in LaTex), default = ['$x$', '']
        self.xlabel = ["$x$", ""]
        # Position of x-axis labels, default = [1.2, -1.2]
        self.xlpos = [1.2, -1.2]
        # Labels for y-axis (in LaTex), default = ['$y$', '']
        self.ylabel = ["$y$", ""]
        # Position of y-axis labels, default = [1.1, -1.1]
        self.ylpos = [1.2, -1.2]
        # Labels for z-axis (in LaTex),
        # default = [r'$\left|1\right>$', r'$\left|0\right>$']
        self.zlabel = [r"$\left|0\right>$", r"$\left|1\right>$"]
        # Position of z-axis labels, default = [1.2, -1.2]
        self.zlpos = [1.2, -1.2]
        # ---font options---
        # Color of fonts, default = 'black'
        self.font_color = plt.rcParams["axes.labelcolor"]
        # Size of fonts, default = 20
        self.font_size = font_size

        # ---vector options---
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000"]
        #: Width of Bloch vectors, default = 5
        self.vector_width = 5
        #: Style of Bloch vectors, default = '-|>' (or 'simple')
        self.vector_style = "-|>"
        #: Sets the width of the vectors arrowhead
        self.vector_mutation = 20

        # ---point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ["b", "r", "g", "#CC6600"]
        # Size of point markers, default = 25
        self.point_size = [25, 32, 35, 45]
        # Shape of point markers, default = ['o','^','d','s']
        self.point_marker = ["o", "s", "d", "^"]

        # ---data lists---
        # Data for point markers
        self.points = []
        # Data for Bloch vectors
        self.vectors = []
        # Data for annotations
        self.annotations = []
        # Number of times sphere has been saved
        self.savenum = 0
        # Style of points, 'm' for multiple colors, 's' for single color
        self.point_style = []

        # status of rendering
        self._rendered = False

    def set_label_convention(self, convention):
        """Set x, y and z labels according to one of conventions.

        Args:
            convention (str):
                One of the following:
                    - "original"
                    - "xyz"
                    - "sx sy sz"
                    - "01"
                    - "polarization jones"
                    - "polarization jones letters"
                    see also: https://en.wikipedia.org/wiki/Jones_calculus
                    - "polarization stokes"
                    see also: https://en.wikipedia.org/wiki/Stokes_parameters
        Raises:
            Exception: If convention is not valid.
        """
        ketex = "$\\left.|%s\\right\\rangle$"
        # \left.| is on purpose, so that every ket has the same size

        if convention == "original":
            self.xlabel = ["$x$", ""]
            self.ylabel = ["$y$", ""]
            self.zlabel = ["$\\left|0\\right>$", "$\\left|1\\right>$"]
        elif convention == "xyz":
            self.xlabel = ["$x$", ""]
            self.ylabel = ["$y$", ""]
            self.zlabel = ["$z$", ""]
        elif convention == "sx sy sz":
            self.xlabel = ["$s_x$", ""]
            self.ylabel = ["$s_y$", ""]
            self.zlabel = ["$s_z$", ""]
        elif convention == "01":
            self.xlabel = ["", ""]
            self.ylabel = ["", ""]
            self.zlabel = ["$\\left|0\\right>$", "$\\left|1\\right>$"]
        elif convention == "polarization jones":
            self.xlabel = [
                ketex % "\\nearrow\\hspace{-1.46}\\swarrow",
                ketex % "\\nwarrow\\hspace{-1.46}\\searrow",
            ]
            self.ylabel = [ketex % "\\circlearrowleft", ketex % "\\circlearrowright"]
            self.zlabel = [ketex % "\\leftrightarrow", ketex % "\\updownarrow"]
        elif convention == "polarization jones letters":
            self.xlabel = [ketex % "D", ketex % "A"]
            self.ylabel = [ketex % "L", ketex % "R"]
            self.zlabel = [ketex % "H", ketex % "V"]
        elif convention == "polarization stokes":
            self.ylabel = [
                "$\\nearrow\\hspace{-1.46}\\swarrow$",
                "$\\nwarrow\\hspace{-1.46}\\searrow$",
            ]
            self.zlabel = ["$\\circlearrowleft$", "$\\circlearrowright$"]
            self.xlabel = ["$\\leftrightarrow$", "$\\updownarrow$"]
        else:
            raise ValueError("No such convention.")

    def __str__(self):
        string = ""
        string += "Bloch data:\n"
        string += "-----------\n"
        string += "Number of points:  " + str(len(self.points)) + "\n"
        string += "Number of vectors: " + str(len(self.vectors)) + "\n"
        string += "\n"
        string += "Bloch sphere properties:\n"
        string += "------------------------\n"
        string += "font_color:      " + str(self.font_color) + "\n"
        string += "font_size:       " + str(self.font_size) + "\n"
        string += "frame_alpha:     " + str(self.frame_alpha) + "\n"
        string += "frame_color:     " + str(self.frame_color) + "\n"
        string += "frame_width:     " + str(self.frame_width) + "\n"
        string += "point_color:     " + str(self.point_color) + "\n"
        string += "point_marker:    " + str(self.point_marker) + "\n"
        string += "point_size:      " + str(self.point_size) + "\n"
        string += "sphere_alpha:    " + str(self.sphere_alpha) + "\n"
        string += "sphere_color:    " + str(self.sphere_color) + "\n"
        string += "figsize:         " + str(self.figsize) + "\n"
        string += "vector_color:    " + str(self.vector_color) + "\n"
        string += "vector_width:    " + str(self.vector_width) + "\n"
        string += "vector_style:    " + str(self.vector_style) + "\n"
        string += "vector_mutation: " + str(self.vector_mutation) + "\n"
        string += "view:            " + str(self.view) + "\n"
        string += "xlabel:          " + str(self.xlabel) + "\n"
        string += "xlpos:           " + str(self.xlpos) + "\n"
        string += "ylabel:          " + str(self.ylabel) + "\n"
        string += "ylpos:           " + str(self.ylpos) + "\n"
        string += "zlabel:          " + str(self.zlabel) + "\n"
        string += "zlpos:           " + str(self.zlpos) + "\n"
        return string

    def clear(self):
        """Resets Bloch sphere data sets to empty."""
        self.points = []
        self.vectors = []
        self.point_style = []
        self.annotations = []

    def add_points(self, points, meth="s"):
        """Add a list of data points to Bloch sphere.

        Args:
            points (array_like):
                Collection of data points.
            meth (str):
                Type of points to plot, use 'm' for multicolored, 'l' for points
                connected with a line.
        """
        if not isinstance(points[0], (list, np.ndarray)):
            points = [[points[0]], [points[1]], [points[2]]]
        points = np.array(points)
        if meth == "s":
            if len(points[0]) == 1:
                pnts = np.array([[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = np.append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.point_style.append("s")
        elif meth == "l":
            self.points.append(points)
            self.point_style.append("l")
        else:
            self.points.append(points)
            self.point_style.append("m")

    def add_vectors(self, vectors):
        """Add a list of vectors to Bloch sphere.

        Args:
            vectors (array_like):
                Array with vectors of unit length or smaller.
        """
        if isinstance(vectors[0], (list, np.ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
        else:
            self.vectors.append(vectors)

    def add_annotation(self, state_or_vector, text, **kwargs):
        """Add a text or LaTeX annotation to Bloch sphere,
        parameterized by a qubit state or a vector.

        Args:
            state_or_vector (array_like):
                Position for the annotation.
                Qobj of a qubit or a vector of 3 elements.
            text (str):
                Annotation text.
                You can use LaTeX, but remember to use raw string
                e.g. r"$\\langle x \\rangle$"
                or escape backslashes
                e.g. "$\\\\langle x \\\\rangle$".
            **kwargs:
                Options as for mplot3d.axes3d.text, including:
                fontsize, color, horizontalalignment, verticalalignment.
        Raises:
            Exception: If input not array_like or tuple.
        """
        if isinstance(state_or_vector, (list, np.ndarray, tuple)) and len(state_or_vector) == 3:
            vec = state_or_vector
        else:
            raise TypeError("Position needs to be specified by a qubit state or a 3D vector.")
        self.annotations.append({"position": vec, "text": text, "opts": kwargs})

    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        self.render()

    def render(self, title=""):
        """
        Render the Bloch sphere and its data sets in on given figure and axes.
        """
        if self._rendered:
            self.axes.clear()

        self._rendered = True

        # Figure instance for Bloch sphere plot
        if not self._ext_fig:
            self.fig = plt.figure(figsize=self.figsize)

        if not self._ext_axes:
            version_match = VERSION_PATTERN_REGEX.search(matplotlib.__version__)
            if tuple(int(x) for x in version_match.group("release").split(".")) >= (3, 4, 0):
                self.axes = Axes3D(
                    self.fig, azim=self.view[0], elev=self.view[1], auto_add_to_figure=False
                )
                self.fig.add_axes(self.axes)
            else:
                self.axes = Axes3D(
                    self.fig,
                    azim=self.view[0],
                    elev=self.view[1],
                )

        if self.background:
            self.axes.clear()
            self.axes.set_xlim3d(-1.3, 1.3)
            self.axes.set_ylim3d(-1.3, 1.3)
            self.axes.set_zlim3d(-1.3, 1.3)
        else:
            self.plot_axes()
            self.axes.set_axis_off()
            self.axes.set_xlim3d(-0.7, 0.7)
            self.axes.set_ylim3d(-0.7, 0.7)
            self.axes.set_zlim3d(-0.7, 0.7)

        # Force aspect ratio
        # MPL 3.2 or previous do not have set_box_aspect
        if hasattr(self.axes, "set_box_aspect"):
            self.axes.set_box_aspect((1, 1, 1))

        self.axes.grid(False)
        self.plot_back()
        self.plot_points()
        self.plot_vectors()
        self.plot_front()
        self.plot_axes_labels()
        self.plot_annotations()
        self.axes.set_title(title, fontsize=self.font_size, y=1.08)

    def plot_back(self):
        """back half of sphere"""
        u_angle = np.linspace(0, np.pi, 25)
        v_angle = np.linspace(0, np.pi, 25)
        x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
        self.axes.plot_surface(
            x_dir,
            y_dir,
            z_dir,
            rstride=2,
            cstride=2,
            color=self.sphere_color,
            linewidth=0,
            alpha=self.sphere_alpha,
        )
        # wireframe
        self.axes.plot_wireframe(
            x_dir,
            y_dir,
            z_dir,
            rstride=5,
            cstride=5,
            color=self.frame_color,
            alpha=self.frame_alpha,
        )
        # equator
        self.axes.plot(
            1.0 * np.cos(u_angle),
            1.0 * np.sin(u_angle),
            zs=0,
            zdir="z",
            lw=self.frame_width,
            color=self.frame_color,
        )
        self.axes.plot(
            1.0 * np.cos(u_angle),
            1.0 * np.sin(u_angle),
            zs=0,
            zdir="x",
            lw=self.frame_width,
            color=self.frame_color,
        )

    def plot_front(self):
        """front half of sphere"""
        u_angle = np.linspace(-np.pi, 0, 25)
        v_angle = np.linspace(0, np.pi, 25)
        x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
        self.axes.plot_surface(
            x_dir,
            y_dir,
            z_dir,
            rstride=2,
            cstride=2,
            color=self.sphere_color,
            linewidth=0,
            alpha=self.sphere_alpha,
        )
        # wireframe
        self.axes.plot_wireframe(
            x_dir,
            y_dir,
            z_dir,
            rstride=5,
            cstride=5,
            color=self.frame_color,
            alpha=self.frame_alpha,
        )
        # equator
        self.axes.plot(
            1.0 * np.cos(u_angle),
            1.0 * np.sin(u_angle),
            zs=0,
            zdir="z",
            lw=self.frame_width,
            color=self.frame_color,
        )
        self.axes.plot(
            1.0 * np.cos(u_angle),
            1.0 * np.sin(u_angle),
            zs=0,
            zdir="x",
            lw=self.frame_width,
            color=self.frame_color,
        )

    def plot_axes(self):
        """axes"""
        span = np.linspace(-1.0, 1.0, 2)
        self.axes.plot(
            span, 0 * span, zs=0, zdir="z", label="X", lw=self.frame_width, color=self.frame_color
        )
        self.axes.plot(
            0 * span, span, zs=0, zdir="z", label="Y", lw=self.frame_width, color=self.frame_color
        )
        self.axes.plot(
            0 * span, span, zs=0, zdir="y", label="Z", lw=self.frame_width, color=self.frame_color
        )

    def plot_axes_labels(self):
        """axes labels"""
        opts = {
            "fontsize": self.font_size,
            "color": self.font_color,
            "horizontalalignment": "center",
            "verticalalignment": "center",
        }
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)

        for item in self.axes.xaxis.get_ticklines() + self.axes.xaxis.get_ticklabels():
            item.set_visible(False)
        for item in self.axes.yaxis.get_ticklines() + self.axes.yaxis.get_ticklabels():
            item.set_visible(False)
        for item in self.axes.zaxis.get_ticklines() + self.axes.zaxis.get_ticklabels():
            item.set_visible(False)

    def plot_vectors(self):
        """Plot vector"""
        # -X and Y data are switched for plotting purposes
        for k, vector in enumerate(self.vectors):

            xs3d = vector[1] * np.array([0, 1])
            ys3d = -vector[0] * np.array([0, 1])
            zs3d = vector[2] * np.array([0, 1])

            color = self.vector_color[np.mod(k, len(self.vector_color))]

            if self.vector_style == "":
                # simple line style
                self.axes.plot(
                    xs3d, ys3d, zs3d, zs=0, zdir="z", label="Z", lw=self.vector_width, color=color
                )
            else:
                # decorated style, with arrow heads
                arr = Arrow3D(
                    xs3d,
                    ys3d,
                    zs3d,
                    mutation_scale=self.vector_mutation,
                    lw=self.vector_width,
                    arrowstyle=self.vector_style,
                    color=color,
                )

                self.axes.add_artist(arr)

    def plot_points(self):
        """Plot points"""
        # -X and Y data are switched for plotting purposes
        for k, point in enumerate(self.points):
            num = len(point[0])
            dist = [
                np.sqrt(point[0][j] ** 2 + point[1][j] ** 2 + point[2][j] ** 2) for j in range(num)
            ]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                # combine arrays so that they can be sorted together
                zipped = list(zip(dist, range(num)))
                zipped.sort()  # sort rates from lowest to highest
                dist, indperm = zip(*zipped)
                indperm = np.array(indperm)
            else:
                indperm = np.arange(num)
            if self.point_style[k] == "s":
                self.axes.scatter(
                    np.real(point[1][indperm]),
                    -np.real(point[0][indperm]),
                    np.real(point[2][indperm]),
                    s=self.point_size[np.mod(k, len(self.point_size))],
                    alpha=1,
                    edgecolor=None,
                    zdir="z",
                    color=self.point_color[np.mod(k, len(self.point_color))],
                    marker=self.point_marker[np.mod(k, len(self.point_marker))],
                )

            elif self.point_style[k] == "m":
                pnt_colors = np.array(self.point_color * math.ceil(num / len(self.point_color)))

                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                marker = self.point_marker[np.mod(k, len(self.point_marker))]
                pnt_size = self.point_size[np.mod(k, len(self.point_size))]
                self.axes.scatter(
                    np.real(point[1][indperm]),
                    -np.real(point[0][indperm]),
                    np.real(point[2][indperm]),
                    s=pnt_size,
                    alpha=1,
                    edgecolor=None,
                    zdir="z",
                    color=pnt_colors,
                    marker=marker,
                )

            elif self.point_style[k] == "l":
                color = self.point_color[np.mod(k, len(self.point_color))]
                self.axes.plot(
                    np.real(point[1]),
                    -np.real(point[0]),
                    np.real(point[2]),
                    alpha=0.75,
                    zdir="z",
                    color=color,
                )

    def plot_annotations(self):
        """Plot annotations"""
        # -X and Y data are switched for plotting purposes
        for annotation in self.annotations:
            vec = annotation["position"]
            opts = {
                "fontsize": self.font_size,
                "color": self.font_color,
                "horizontalalignment": "center",
                "verticalalignment": "center",
            }
            opts.update(annotation["opts"])
            self.axes.text(vec[1], -vec[0], vec[2], annotation["text"], **opts)

    def show(self, title=""):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(title=title)
        if self.fig:
            plt.show()

    def save(self, name=None, output="png", dirc=None):
        """Saves Bloch sphere to file of type ``format`` in directory ``dirc``.

        Args:
            name (str):
                Name of saved image. Must include path and format as well.
                i.e. '/Users/Paul/Desktop/bloch.png'
                This overrides the 'format' and 'dirc' arguments.
            output (str):
                Format of output image.
            dirc (str):
                Directory for output images. Defaults to current working directory.
        """

        self.render()
        if dirc:
            if not os.path.isdir(os.getcwd() + "/" + str(dirc)):
                os.makedirs(os.getcwd() + "/" + str(dirc))
        if name is None:
            if dirc:
                self.fig.savefig(
                    os.getcwd() + "/" + str(dirc) + "/bloch_" + str(self.savenum) + "." + output
                )
            else:
                self.fig.savefig(os.getcwd() + "/bloch_" + str(self.savenum) + "." + output)
        else:
            self.fig.savefig(name)
        self.savenum += 1
        if self.fig:
            matplotlib_close_if_inline(self.fig)


def _hide_tick_lines_and_labels(axis):
    """
    Set visible property of ticklines and ticklabels of an axis to False
    """
    for item in axis.get_ticklines() + axis.get_ticklabels():
        item.set_visible(False)
