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
Visualization function for animation of state transitions by applying gates to single qubit.
"""
import sys
from math import sin, cos, acos, sqrt
import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils.deprecation import deprecate_func


def _normalize(v, tolerance=0.00001):
    """Makes sure magnitude of the vector is 1 with given tolerance"""

    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


class _Quaternion:
    """For calculating vectors on unit sphere"""

    def __init__(self):
        self._val = None

    @staticmethod
    def from_axisangle(theta, v):
        """Create quaternion from axis"""
        v = _normalize(v)

        new_quaternion = _Quaternion()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def from_value(value):
        """Create quaternion from vector"""
        new_quaternion = _Quaternion()
        new_quaternion._val = value
        return new_quaternion

    def _axisangle_to_q(self, theta, v):
        """Convert axis and angle to quaternion"""
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta / 2.0)
        x = x * sin(theta / 2.0)
        y = y * sin(theta / 2.0)
        z = z * sin(theta / 2.0)

        self._val = np.array([w, x, y, z])

    def __mul__(self, b):
        """Multiplication of quaternion with quaternion or vector"""

        if isinstance(b, _Quaternion):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                raise ValueError(f"Input vector has invalid length {len(b)}")
            return self._multiply_with_vector(b)
        else:
            return NotImplemented

    def _multiply_with_quaternion(self, q_2):
        """Multiplication of quaternion with quaternion"""
        w_1, x_1, y_1, z_1 = self._val
        w_2, x_2, y_2, z_2 = q_2._val
        w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
        z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2

        result = _Quaternion.from_value(np.array((w, x, y, z)))
        return result

    def _multiply_with_vector(self, v):
        """Multiplication of quaternion with vector"""
        q_2 = _Quaternion.from_value(np.append((0.0), v))
        return (self * q_2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        """Conjugation of quaternion"""
        w, x, y, z = self._val
        result = _Quaternion.from_value(np.array((w, -x, -y, -z)))
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return f"(({theta}; {v[0]}, {v[1]}, {v[2]}))"

    def get_axisangle(self):
        """Returns angle and vector of quaternion"""
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, _normalize(v)

    def tolist(self):
        """Converts quaternion to a list"""
        return self._val.tolist()

    def vector_norm(self):
        """Calculates norm of quaternion"""
        _, v = self.get_axisangle()
        return np.linalg.norm(v)


@deprecate_func(
    since="1.2.0",
    removal_timeline="in the 2.0 release",
)
def visualize_transition(circuit, trace=False, saveas=None, fpg=100, spg=2):
    """
    Creates animation showing transitions between states of a single
    qubit by applying quantum gates.

    Args:
        circuit (QuantumCircuit): Qiskit single-qubit QuantumCircuit. Gates supported are
            h,x, y, z, rx, ry, rz, s, sdg, t, tdg and u1.
        trace (bool): Controls whether to display tracing vectors - history of 10 past vectors
            at each step of the animation.
        saveas (str): User can choose to save the animation as a video to their filesystem.
            This argument is a string of path with filename and extension (e.g. "movie.mp4" to
            save the video in current working directory).
        fpg (int): Frames per gate. Finer control over animation smoothness and computational
            needs to render the animation. Works well for tkinter GUI as it is, for jupyter GUI
            it might be preferable to choose fpg between 5-30.
        spg (int): Seconds per gate. How many seconds should animation of individual gate
            transitions take.

    Returns:
        IPython.core.display.HTML:
            If arg jupyter is set to True. Otherwise opens tkinter GUI and returns
            after the GUI is closed.

    Raises:
        MissingOptionalLibraryError: Must have Matplotlib (and/or IPython) installed.
        VisualizationError: Given gate(s) are not supported.

    """
    try:
        from IPython.display import HTML

        has_ipython = True
    except ImportError:
        has_ipython = False

    try:
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D
        from .bloch import Bloch
        from .exceptions import VisualizationError

        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    jupyter = False
    if ("ipykernel" in sys.modules) and ("spyder" not in sys.modules):
        jupyter = True

    if not has_matplotlib:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="visualize_transition",
            pip_install="pip install matplotlib",
        )
    if not has_ipython and jupyter is True:
        raise MissingOptionalLibraryError(
            libname="IPython",
            name="visualize_transition",
            pip_install="pip install ipython",
        )
    if len(circuit.qubits) != 1:
        raise VisualizationError("Only one qubit circuits are supported")

    frames_per_gate = fpg
    time_between_frames = (spg * 1000) / fpg

    # quaternions of gates which don't take parameters
    simple_gates = {}
    simple_gates["x"] = (
        "x",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [1, 0, 0]),
        "#1abc9c",
    )
    simple_gates["y"] = (
        "y",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 1, 0]),
        "#2ecc71",
    )
    simple_gates["z"] = (
        "z",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 0, 1]),
        "#3498db",
    )
    simple_gates["s"] = (
        "s",
        _Quaternion.from_axisangle(np.pi / 2 / frames_per_gate, [0, 0, 1]),
        "#9b59b6",
    )
    simple_gates["sdg"] = (
        "sdg",
        _Quaternion.from_axisangle(-np.pi / 2 / frames_per_gate, [0, 0, 1]),
        "#8e44ad",
    )
    simple_gates["h"] = (
        "h",
        _Quaternion.from_axisangle(np.pi / frames_per_gate, _normalize([1, 0, 1])),
        "#34495e",
    )
    simple_gates["t"] = (
        "t",
        _Quaternion.from_axisangle(np.pi / 4 / frames_per_gate, [0, 0, 1]),
        "#e74c3c",
    )
    simple_gates["tdg"] = (
        "tdg",
        _Quaternion.from_axisangle(-np.pi / 4 / frames_per_gate, [0, 0, 1]),
        "#c0392b",
    )

    list_of_circuit_gates = []

    for gate, _, _ in circuit._data:
        if gate.name == "barrier":
            continue
        if gate.name in simple_gates:
            list_of_circuit_gates.append(simple_gates[gate.name])
        elif gate.name == "rx":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [1, 0, 0])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#16a085"))
        elif gate.name == "ry":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 1, 0])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#27ae60"))
        elif gate.name == "rz":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 0, 1])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#2980b9"))
        elif gate.name == "u1":
            theta = gate.params[0]
            quaternion = _Quaternion.from_axisangle(theta / frames_per_gate, [0, 0, 1])
            list_of_circuit_gates.append((f"{gate.name}: {theta:.2f}", quaternion, "#f1c40f"))
        else:
            raise VisualizationError(f"Gate {gate.name} is not supported")

    if len(list_of_circuit_gates) == 0:
        raise VisualizationError("Nothing to visualize.")

    starting_pos = _normalize(np.array([0, 0, 1]))

    fig = plt.figure(figsize=(5, 5))
    if tuple(int(x) for x in matplotlib.__version__.split(".")) >= (3, 4, 0):
        _ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(_ax)
    else:
        _ax = Axes3D(fig)

    _ax.set_xlim(-10, 10)
    _ax.set_ylim(-10, 10)
    sphere = Bloch(axes=_ax)

    class Namespace:
        """Helper class serving as scope container"""

        def __init__(self):
            self.new_vec = []
            self.last_gate = -2
            self.colors = []
            self.pnts = []

    namespace = Namespace()
    namespace.new_vec = starting_pos

    def animate(i):
        sphere.clear()

        # starting gate count from -1 which is the initial vector
        gate_counter = (i - 1) // frames_per_gate
        if gate_counter != namespace.last_gate:
            namespace.pnts.append([[], [], []])
            namespace.colors.append(list_of_circuit_gates[gate_counter][2])

        # starts with default vector [0,0,1]
        if i == 0:
            sphere.add_vectors(namespace.new_vec)
            namespace.pnts[0][0].append(namespace.new_vec[0])
            namespace.pnts[0][1].append(namespace.new_vec[1])
            namespace.pnts[0][2].append(namespace.new_vec[2])
            namespace.colors[0] = "r"
            sphere.make_sphere()
            return _ax

        namespace.new_vec = list_of_circuit_gates[gate_counter][1] * namespace.new_vec

        namespace.pnts[gate_counter + 1][0].append(namespace.new_vec[0])
        namespace.pnts[gate_counter + 1][1].append(namespace.new_vec[1])
        namespace.pnts[gate_counter + 1][2].append(namespace.new_vec[2])

        sphere.add_vectors(namespace.new_vec)
        if trace:
            # sphere.add_vectors(namespace.points)
            for point_set in namespace.pnts:
                sphere.add_points([point_set[0], point_set[1], point_set[2]])

        sphere.vector_color = [list_of_circuit_gates[gate_counter][2]]
        sphere.point_color = namespace.colors
        sphere.point_marker = "o"

        annotation_text = list_of_circuit_gates[gate_counter][0]
        annotationvector = [1.40, -0.45, 1.65]
        sphere.add_annotation(
            annotationvector,
            annotation_text,
            color=list_of_circuit_gates[gate_counter][2],
            fontsize=30,
            horizontalalignment="left",
        )

        sphere.make_sphere()

        namespace.last_gate = gate_counter
        return _ax

    def init():
        sphere.vector_color = ["r"]
        return _ax

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(frames_per_gate * len(list_of_circuit_gates) + 1),
        init_func=init,
        blit=False,
        repeat=False,
        interval=time_between_frames,
    )

    if saveas:
        ani.save(saveas, fps=30)
    if jupyter:
        # This is necessary to overcome matplotlib memory limit
        matplotlib.rcParams["animation.embed_limit"] = 50
        plt.close(fig)
        return HTML(ani.to_jshtml())
    plt.show()
    plt.close(fig)
    return None
