# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A library of known embodiments of RXXGate in terms of other gates,
for some generic or specific angles.

TODO: discover these automatically from the gates' algebraic definition
"""

from __future__ import annotations
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import (
    RXXGate,
    RZZGate,
    RZXGate,
    RYYGate,
    CRZGate,
    CRXGate,
    CRYGate,
    CPhaseGate,
    CZGate,
    CXGate,
    CYGate,
    CHGate,
    ECRGate,
)


rxx_circuit = QuantumCircuit(2)
theta = Parameter("θ")
rxx_circuit.rxx(theta, 0, 1)

rzz_circuit = QuantumCircuit(2)
theta = Parameter("θ")
rzz_circuit.h(0)
rzz_circuit.h(1)
rzz_circuit.rzz(theta, 0, 1)
rzz_circuit.h(0)
rzz_circuit.h(1)

rzx_circuit = QuantumCircuit(2)
rzx_circuit.h(0)
rzx_circuit.rzx(theta, 0, 1)
rzx_circuit.h(0)

ryy_circuit = QuantumCircuit(2)
ryy_circuit.s(0)
ryy_circuit.s(1)
ryy_circuit.ryy(theta, 0, 1)
ryy_circuit.sdg(0)
ryy_circuit.sdg(1)

cphase_circuit = QuantumCircuit(2)
cphase_circuit.h(0)
cphase_circuit.h(1)
cphase_circuit.cp(-2 * theta, 0, 1)
cphase_circuit.rz(theta, 0)
cphase_circuit.rz(theta, 1)
cphase_circuit.h(0)
cphase_circuit.h(1)
cphase_circuit.global_phase += theta / 2

crz_circuit = QuantumCircuit(2)
crz_circuit.h(0)
crz_circuit.h(1)
crz_circuit.crz(-2 * theta, 0, 1)
crz_circuit.rz(theta, 1)
crz_circuit.h(0)
crz_circuit.h(1)

crx_circuit = QuantumCircuit(2)
crx_circuit.h(0)
crx_circuit.crx(-2 * theta, 0, 1)
crx_circuit.rx(theta, 1)
crx_circuit.h(0)

cry_circuit = QuantumCircuit(2)
cry_circuit.h(0)
cry_circuit.s(1)
cry_circuit.cry(-2 * theta, 0, 1)
cry_circuit.ry(theta, 1)
cry_circuit.h(0)
cry_circuit.sdg(1)

cz_circuit = QuantumCircuit(2)
cz_circuit.h(0)
cz_circuit.h(1)
cz_circuit.cz(0, 1)
cz_circuit.s(0)
cz_circuit.s(1)
cz_circuit.h(0)
cz_circuit.h(1)
cz_circuit.global_phase -= np.pi / 4

cx_circuit = QuantumCircuit(2)
cx_circuit.h(0)
cx_circuit.cx(0, 1)
cx_circuit.s(0)
cx_circuit.sx(1)
cx_circuit.h(0)
cx_circuit.global_phase -= np.pi / 4

cy_circuit = QuantumCircuit(2)
cy_circuit.h(0)
cy_circuit.s(1)
cy_circuit.cy(0, 1)
cy_circuit.s(0)
cy_circuit.sdg(1)
cy_circuit.sx(1)
cy_circuit.h(0)
cy_circuit.global_phase -= np.pi / 4

ch_circuit = QuantumCircuit(2)
ch_circuit.h(0)
ch_circuit.tdg(1)
ch_circuit.h(1)
ch_circuit.sdg(1)
ch_circuit.ch(0, 1)
ch_circuit.s(0)
ch_circuit.s(1)
ch_circuit.h(1)
ch_circuit.t(1)
ch_circuit.sx(1)
ch_circuit.h(0)
ch_circuit.global_phase -= np.pi / 4

ecr_circuit = QuantumCircuit(2)
ecr_circuit.h(0)
ecr_circuit.s(0)
ecr_circuit.x(0)
ecr_circuit.x(1)
ecr_circuit.ecr(0, 1)
ecr_circuit.s(0)
ecr_circuit.h(0)
ecr_circuit.global_phase -= np.pi / 2

XXEmbodiments = {
    RXXGate: rxx_circuit,
    RYYGate: ryy_circuit,
    RZZGate: rzz_circuit,
    RZXGate: rzx_circuit,
    CRXGate: crx_circuit,
    CRYGate: cry_circuit,
    CRZGate: crz_circuit,
    CPhaseGate: cphase_circuit,
    CXGate: cx_circuit,
    CYGate: cy_circuit,
    CZGate: cz_circuit,
    CHGate: ch_circuit,
    ECRGate: ecr_circuit,
}
