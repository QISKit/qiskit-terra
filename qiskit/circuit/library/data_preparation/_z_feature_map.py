# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Create a new first-order Pauli-Z expansion circuit."""


from collections.abc import Callable
import numpy as np
from qiskit.utils.deprecation import deprecate_func

from .pauli_feature_map import PauliFeatureMap


class ZFeatureMap(PauliFeatureMap):
    """The first order Pauli Z-evolution circuit.

    On 3 qubits and with 2 repetitions the circuit is represented by:

    .. code-block:: text

        ┌───┐┌─────────────┐┌───┐┌─────────────┐
        ┤ H ├┤ P(2.0*x[0]) ├┤ H ├┤ P(2.0*x[0]) ├
        ├───┤├─────────────┤├───┤├─────────────┤
        ┤ H ├┤ P(2.0*x[1]) ├┤ H ├┤ P(2.0*x[1]) ├
        ├───┤├─────────────┤├───┤├─────────────┤
        ┤ H ├┤ P(2.0*x[2]) ├┤ H ├┤ P(2.0*x[2]) ├
        └───┘└─────────────┘└───┘└─────────────┘

    This is a sub-class of :class:`~qiskit.circuit.library.PauliFeatureMap` where the Pauli
    strings are fixed as `['Z']`. As a result the first order expansion will be a circuit without
    entangling gates.

    Examples:

        >>> prep = ZFeatureMap(3, reps=3, insert_barriers=True)
        >>> print(prep.decompose())
             ┌───┐ ░ ┌─────────────┐ ░ ┌───┐ ░ ┌─────────────┐ ░ ┌───┐ ░ ┌─────────────┐
        q_0: ┤ H ├─░─┤ P(2.0*x[0]) ├─░─┤ H ├─░─┤ P(2.0*x[0]) ├─░─┤ H ├─░─┤ P(2.0*x[0]) ├
             ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤
        q_1: ┤ H ├─░─┤ P(2.0*x[1]) ├─░─┤ H ├─░─┤ P(2.0*x[1]) ├─░─┤ H ├─░─┤ P(2.0*x[1]) ├
             ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤
        q_2: ┤ H ├─░─┤ P(2.0*x[2]) ├─░─┤ H ├─░─┤ P(2.0*x[2]) ├─░─┤ H ├─░─┤ P(2.0*x[2]) ├
             └───┘ ░ └─────────────┘ ░ └───┘ ░ └─────────────┘ ░ └───┘ ░ └─────────────┘

        >>> data_map = lambda x: x[0]*x[0] + 1  # note: input is an array
        >>> prep = ZFeatureMap(3, reps=1, data_map_func=data_map)
        >>> print(prep.decompose())
             ┌───┐┌──────────────────────┐
        q_0: ┤ H ├┤ P(2.0*x[0]**2 + 2.0) ├
             ├───┤├──────────────────────┤
        q_1: ┤ H ├┤ P(2.0*x[1]**2 + 2.0) ├
             ├───┤├──────────────────────┤
        q_2: ┤ H ├┤ P(2.0*x[2]**2 + 2.0) ├
             └───┘└──────────────────────┘

        >>> from qiskit.circuit.library import TwoLocal
        >>> ry = TwoLocal(3, "ry", "cz", reps=1)
        >>> classifier = ZFeatureMap(3, reps=1) + ry
        >>> print(classifier.decompose())
             ┌───┐┌─────────────┐┌──────────┐      ┌──────────┐
        q_0: ┤ H ├┤ P(2.0*x[0]) ├┤ RY(θ[0]) ├─■──■─┤ RY(θ[3]) ├────────────
             ├───┤├─────────────┤├──────────┤ │  │ └──────────┘┌──────────┐
        q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ RY(θ[1]) ├─■──┼──────■──────┤ RY(θ[4]) ├
             ├───┤├─────────────┤├──────────┤    │      │      ├──────────┤
        q_2: ┤ H ├┤ P(2.0*x[2]) ├┤ RY(θ[2]) ├────■──────■──────┤ RY(θ[5]) ├
             └───┘└─────────────┘└──────────┘                  └──────────┘

    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "Use the z_feature_map function as a replacement. Note that this will no longer "
            "return a BlueprintCircuit, but just a plain QuantumCircuit."
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        data_map_func: Callable[[np.ndarray], float] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZFeatureMap",
    ) -> None:
        """
        Args:
            feature_dimension: The number of features
            reps: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
            name: Name of the circuit.

        """
        super().__init__(
            feature_dimension=feature_dimension,
            paulis=["Z"],
            reps=reps,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
