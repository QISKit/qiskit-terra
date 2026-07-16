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

"""
Standard gates
"""

from .h import HGate, CHGate
from .i import IGate
from .p import PhaseGate, CPhaseGate, MCPhaseGate
from .r import RGate
from .rx import RXGate, CRXGate
from .rxx import RXXGate
from .ry import RYGate, CRYGate
from .ryy import RYYGate
from .rz import RZGate, CRZGate
from .rzz import RZZGate
from .rzx import RZXGate
from .xx_minus_yy import XXMinusYYGate
from .xx_plus_yy import XXPlusYYGate
from .ecr import ECRGate
from .s import SGate, SdgGate, CSGate, CSdgGate
from .swap import SwapGate, CSwapGate
from .iswap import iSwapGate
from .sx import SXGate, SXdgGate, CSXGate
from .dcx import DCXGate
from .t import TGate, TdgGate
from .u import UGate, CUGate
from .u1 import U1Gate, CU1Gate, MCU1Gate
from .u2 import U2Gate
from .u3 import U3Gate, CU3Gate
from .x import XGate, CXGate, CCXGate, C3XGate, C3SXGate, C4XGate, RCCXGate, RC3XGate
from .x import MCXGate, MCXGrayCode, MCXRecursive, MCXVChain
from .y import YGate, CYGate
from .z import ZGate, CZGate, CCZGate
from .global_phase import GlobalPhaseGate


__all__ = [
    "C3SXGate",
    "C3XGate",
    "C4XGate",
    "CCXGate",
    "CCZGate",
    "CHGate",
    "CPhaseGate",
    "CRXGate",
    "CRYGate",
    "CRZGate",
    "CSGate",
    "CSXGate",
    "CSdgGate",
    "CSwapGate",
    "CU1Gate",
    "CU3Gate",
    "CUGate",
    "CXGate",
    "CYGate",
    "CZGate",
    "DCXGate",
    "ECRGate",
    "GlobalPhaseGate",
    "HGate",
    "IGate",
    "MCPhaseGate",
    "MCU1Gate",
    "MCXGate",
    "MCXGrayCode",
    "MCXRecursive",
    "MCXVChain",
    "PhaseGate",
    "RC3XGate",
    "RCCXGate",
    "RGate",
    "RXGate",
    "RXXGate",
    "RYGate",
    "RYYGate",
    "RZGate",
    "RZXGate",
    "RZZGate",
    "SGate",
    "SXGate",
    "SXdgGate",
    "SdgGate",
    "SwapGate",
    "TGate",
    "TdgGate",
    "U1Gate",
    "U2Gate",
    "U3Gate",
    "UGate",
    "XGate",
    "XXMinusYYGate",
    "XXPlusYYGate",
    "YGate",
    "ZGate",
    "get_standard_gate_name_mapping",
    "iSwapGate",
]


def get_standard_gate_name_mapping():
    """Return a dictionary mapping the name of standard gates and instructions to an object for
    that name.

    Examples:

        .. code-block:: python

            from qiskit.circuit.library import get_standard_gate_name_mapping

            gate_name_map = get_standard_gate_name_mapping()
            cx_object = gate_name_map["cx"]

            print(cx_object)
            print(type(cx_object))

        .. code-block:: text

            Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])
            _SingletonCXGate
    """

    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.measure import Measure
    from qiskit.circuit.delay import Delay
    from qiskit.circuit.reset import Reset

    lambda_ = Parameter("λ")
    theta = Parameter("ϴ")
    phi = Parameter("φ")
    gamma = Parameter("γ")
    beta = Parameter("β")
    time = Parameter("t")

    # Standard gates library mapping, multicontrolled gates not included since they're
    # variable width
    gates = [
        IGate(),
        SXGate(),
        XGate(),
        CXGate(),
        RZGate(lambda_),
        RGate(theta, phi),
        C3SXGate(),
        CCXGate(),
        DCXGate(),
        CHGate(),
        CPhaseGate(theta),
        CRXGate(theta),
        CRYGate(theta),
        CRZGate(theta),
        CSwapGate(),
        CSXGate(),
        CUGate(theta, phi, lambda_, gamma),
        CU1Gate(lambda_),
        CU3Gate(theta, phi, lambda_),
        CYGate(),
        CZGate(),
        CCZGate(),
        GlobalPhaseGate(theta),
        HGate(),
        PhaseGate(theta),
        RCCXGate(),
        RC3XGate(),
        RXGate(theta),
        RXXGate(theta),
        RYGate(theta),
        RYYGate(theta),
        RZZGate(theta),
        RZXGate(theta),
        XXMinusYYGate(theta, beta),
        XXPlusYYGate(theta, beta),
        ECRGate(),
        SGate(),
        SdgGate(),
        CSGate(),
        CSdgGate(),
        SwapGate(),
        iSwapGate(),
        SXdgGate(),
        TGate(),
        TdgGate(),
        UGate(theta, phi, lambda_),
        U1Gate(lambda_),
        U2Gate(phi, lambda_),
        U3Gate(theta, phi, lambda_),
        YGate(),
        ZGate(),
        Delay(time),
        Reset(),
        Measure(),
    ]
    name_mapping = {gate.name: gate for gate in gates}
    return name_mapping
