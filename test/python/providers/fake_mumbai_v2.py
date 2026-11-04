# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Mock BackendV2 object without run implemented for testing backwards compat"""

import datetime

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.circuit.library.standard_gates import (
    CXGate,
    RZXGate,
    XGate,
    SXGate,
    RZGate,
)
from qiskit.providers.backend import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties


class FakeMumbaiFractionalCX(BackendV2):
    """A fake mumbai backend."""

    def __init__(self):
        super().__init__(
            name="FakeMumbaiFractionalCX",
            description="A fake BackendV2 example based on IBM Mumbai",
            online_date=datetime.datetime.now(datetime.UTC),
            backend_version="0.0.1",
        )
        dt = 0.2222222222222222e-9
        self._target = Target(dt=dt)
        self._phi = Parameter("phi")
        rz_props = {
            (0,): InstructionProperties(duration=0.0, error=0),
            (1,): InstructionProperties(duration=0.0, error=0),
            (2,): InstructionProperties(duration=0.0, error=0),
            (3,): InstructionProperties(duration=0.0, error=0),
            (4,): InstructionProperties(duration=0.0, error=0),
            (5,): InstructionProperties(duration=0.0, error=0),
            (6,): InstructionProperties(duration=0.0, error=0),
            (7,): InstructionProperties(duration=0.0, error=0),
            (8,): InstructionProperties(duration=0.0, error=0),
            (9,): InstructionProperties(duration=0.0, error=0),
            (10,): InstructionProperties(duration=0.0, error=0),
            (11,): InstructionProperties(duration=0.0, error=0),
            (12,): InstructionProperties(duration=0.0, error=0),
            (13,): InstructionProperties(duration=0.0, error=0),
            (14,): InstructionProperties(duration=0.0, error=0),
            (15,): InstructionProperties(duration=0.0, error=0),
            (16,): InstructionProperties(duration=0.0, error=0),
            (17,): InstructionProperties(duration=0.0, error=0),
            (18,): InstructionProperties(duration=0.0, error=0),
            (19,): InstructionProperties(duration=0.0, error=0),
            (20,): InstructionProperties(duration=0.0, error=0),
            (21,): InstructionProperties(duration=0.0, error=0),
            (22,): InstructionProperties(duration=0.0, error=0),
            (23,): InstructionProperties(duration=0.0, error=0),
            (24,): InstructionProperties(duration=0.0, error=0),
            (25,): InstructionProperties(duration=0.0, error=0),
            (26,): InstructionProperties(duration=0.0, error=0),
        }
        self._target.add_instruction(RZGate(self._phi), rz_props)
        x_props = {
            (0,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00020056469709026198
            ),
            (1,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0004387432040599484
            ),
            (2,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002196765027963209
            ),
            (3,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0003065541555566093
            ),
            (4,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002402026686478811
            ),
            (5,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002162777062721698
            ),
            (6,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00021981280474256117
            ),
            (7,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00018585647396926756
            ),
            (8,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00027053333211825124
            ),
            (9,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002603116226593832
            ),
            (10,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00023827406030798066
            ),
            (11,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024856063217108685
            ),
            (12,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002065075637361354
            ),
            (13,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024898181450337464
            ),
            (14,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00017758796319636606
            ),
            (15,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00016530893922883836
            ),
            (16,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0003213658218204255
            ),
            (17,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024068450432012685
            ),
            (18,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00026676441863976344
            ),
            (19,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00017090891698571018
            ),
            (20,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00021057196071004095
            ),
            (21,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00030445404779882887
            ),
            (22,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00019322295843406375
            ),
            (23,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00030966037392287727
            ),
            (24,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00023570754161126
            ),
            (25,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00018367783963229033
            ),
            (26,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00019630609928571516
            ),
        }
        self._target.add_instruction(XGate(), x_props)
        sx_props = {
            (0,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00020056469709026198
            ),
            (1,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0004387432040599484
            ),
            (2,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002196765027963209
            ),
            (3,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0003065541555566093
            ),
            (4,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002402026686478811
            ),
            (5,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002162777062721698
            ),
            (6,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00021981280474256117
            ),
            (7,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00018585647396926756
            ),
            (8,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00027053333211825124
            ),
            (9,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002603116226593832
            ),
            (10,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00023827406030798066
            ),
            (11,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024856063217108685
            ),
            (12,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0002065075637361354
            ),
            (13,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024898181450337464
            ),
            (14,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00017758796319636606
            ),
            (15,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00016530893922883836
            ),
            (16,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.0003213658218204255
            ),
            (17,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00024068450432012685
            ),
            (18,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00026676441863976344
            ),
            (19,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00017090891698571018
            ),
            (20,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00021057196071004095
            ),
            (21,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00030445404779882887
            ),
            (22,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00019322295843406375
            ),
            (23,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00030966037392287727
            ),
            (24,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00023570754161126
            ),
            (25,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00018367783963229033
            ),
            (26,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00019630609928571516
            ),
        }
        self._target.add_instruction(SXGate(), sx_props)
        chunk_size = 16
        cx_props = {
            (0, 1): InstructionProperties(
                duration=101 * chunk_size * dt, error=0.030671121181161276
            ),
            (4, 1): InstructionProperties(
                duration=70 * chunk_size * dt, error=0.014041986073052737
            ),
            (4, 7): InstructionProperties(duration=74 * chunk_size * dt, error=0.0052040275323747),
            (10, 7): InstructionProperties(
                duration=92 * chunk_size * dt, error=0.005625282141655502
            ),
            (10, 12): InstructionProperties(
                duration=84 * chunk_size * dt, error=0.005771827440726435
            ),
            (15, 12): InstructionProperties(
                duration=84 * chunk_size * dt, error=0.0050335609425562755
            ),
            (15, 18): InstructionProperties(
                duration=64 * chunk_size * dt, error=0.0051374141171115495
            ),
            (12, 13): InstructionProperties(
                duration=70 * chunk_size * dt, error=0.011361175954064051
            ),
            (13, 14): InstructionProperties(
                duration=101 * chunk_size * dt, error=0.005231334872256355
            ),
            # From FakeMumbai:
            (1, 0): InstructionProperties(
                duration=4.551111111111111e-07, error=0.030671121181161276
            ),
            (1, 2): InstructionProperties(
                duration=7.395555555555556e-07, error=0.03420291964205785
            ),
            (1, 4): InstructionProperties(
                duration=3.6266666666666663e-07, error=0.014041986073052737
            ),
            (2, 1): InstructionProperties(duration=7.04e-07, error=0.03420291964205785),
            (2, 3): InstructionProperties(
                duration=4.266666666666666e-07, error=0.005618162036535312
            ),
            (3, 2): InstructionProperties(
                duration=3.911111111111111e-07, error=0.005618162036535312
            ),
            (3, 5): InstructionProperties(
                duration=3.5555555555555553e-07, error=0.006954580732294352
            ),
            (5, 3): InstructionProperties(
                duration=3.911111111111111e-07, error=0.006954580732294352
            ),
            (5, 8): InstructionProperties(
                duration=1.3155555555555553e-06, error=0.021905829471073668
            ),
            (6, 7): InstructionProperties(
                duration=2.4177777777777775e-07, error=0.011018069718028878
            ),
            (7, 4): InstructionProperties(
                duration=3.7688888888888884e-07, error=0.0052040275323747
            ),
            (7, 6): InstructionProperties(
                duration=2.7733333333333333e-07, error=0.011018069718028878
            ),
            (7, 10): InstructionProperties(
                duration=4.337777777777778e-07, error=0.005625282141655502
            ),
            (8, 5): InstructionProperties(
                duration=1.351111111111111e-06, error=0.021905829471073668
            ),
            (8, 9): InstructionProperties(
                duration=6.897777777777777e-07, error=0.011889378687341773
            ),
            (8, 11): InstructionProperties(
                duration=5.902222222222222e-07, error=0.009523844852027258
            ),
            (9, 8): InstructionProperties(
                duration=6.542222222222222e-07, error=0.011889378687341773
            ),
            (11, 8): InstructionProperties(
                duration=6.257777777777777e-07, error=0.009523844852027258
            ),
            (11, 14): InstructionProperties(
                duration=4.053333333333333e-07, error=0.004685421425282804
            ),
            (12, 10): InstructionProperties(
                duration=3.9822222222222215e-07, error=0.005771827440726435
            ),
            (12, 15): InstructionProperties(
                duration=4.053333333333333e-07, error=0.0050335609425562755
            ),
            (13, 12): InstructionProperties(
                duration=5.831111111111111e-07, error=0.011361175954064051
            ),
            (14, 11): InstructionProperties(
                duration=3.697777777777778e-07, error=0.004685421425282804
            ),
            (14, 13): InstructionProperties(
                duration=3.5555555555555553e-07, error=0.005231334872256355
            ),
            (14, 16): InstructionProperties(
                duration=3.484444444444444e-07, error=0.0051117141032224755
            ),
            (16, 14): InstructionProperties(
                duration=3.1288888888888885e-07, error=0.0051117141032224755
            ),
            (16, 19): InstructionProperties(
                duration=7.537777777777777e-07, error=0.013736796355458464
            ),
            (17, 18): InstructionProperties(
                duration=2.488888888888889e-07, error=0.007267536233537236
            ),
            (18, 15): InstructionProperties(
                duration=3.413333333333333e-07, error=0.0051374141171115495
            ),
            (18, 17): InstructionProperties(
                duration=2.8444444444444443e-07, error=0.007267536233537236
            ),
            (18, 21): InstructionProperties(
                duration=4.977777777777778e-07, error=0.007718304749257138
            ),
            (19, 16): InstructionProperties(
                duration=7.182222222222222e-07, error=0.013736796355458464
            ),
            (19, 20): InstructionProperties(
                duration=4.266666666666666e-07, error=0.005757038521092134
            ),
            (19, 22): InstructionProperties(
                duration=3.6266666666666663e-07, error=0.004661878013991871
            ),
            (20, 19): InstructionProperties(
                duration=3.911111111111111e-07, error=0.005757038521092134
            ),
            (21, 18): InstructionProperties(
                duration=5.333333333333332e-07, error=0.007718304749257138
            ),
            (21, 23): InstructionProperties(
                duration=3.911111111111111e-07, error=0.007542515578725928
            ),
            (22, 19): InstructionProperties(
                duration=3.271111111111111e-07, error=0.004661878013991871
            ),
            (22, 25): InstructionProperties(
                duration=4.835555555555555e-07, error=0.005536735115231589
            ),
            (23, 21): InstructionProperties(
                duration=4.266666666666666e-07, error=0.007542515578725928
            ),
            (23, 24): InstructionProperties(
                duration=6.613333333333332e-07, error=0.010797784688907186
            ),
            (24, 23): InstructionProperties(
                duration=6.257777777777777e-07, error=0.010797784688907186
            ),
            (24, 25): InstructionProperties(
                duration=4.337777777777778e-07, error=0.006127506135155392
            ),
            (25, 22): InstructionProperties(duration=4.48e-07, error=0.005536735115231589),
            (25, 24): InstructionProperties(
                duration=4.693333333333333e-07, error=0.006127506135155392
            ),
            (25, 26): InstructionProperties(
                duration=3.484444444444444e-07, error=0.0048451525929122385
            ),
            (26, 25): InstructionProperties(
                duration=3.1288888888888885e-07, error=0.0048451525929122385
            ),
        }
        self.target.add_instruction(CXGate(), cx_props)
        # Error and duration the same as CX
        rzx_90_props = {
            (0, 1): InstructionProperties(
                duration=101 * chunk_size * dt, error=0.030671121181161276
            ),
            (4, 1): InstructionProperties(
                duration=70 * chunk_size * dt, error=0.014041986073052737
            ),
            (4, 7): InstructionProperties(duration=74 * chunk_size * dt, error=0.0052040275323747),
            (10, 7): InstructionProperties(
                duration=92 * chunk_size * dt, error=0.005625282141655502
            ),
            (10, 12): InstructionProperties(
                duration=84 * chunk_size * dt, error=0.005771827440726435
            ),
            (15, 12): InstructionProperties(
                duration=84 * chunk_size * dt, error=0.0050335609425562755
            ),
            (15, 18): InstructionProperties(
                duration=64 * chunk_size * dt, error=0.0051374141171115495
            ),
            (12, 13): InstructionProperties(
                duration=70 * chunk_size * dt, error=0.011361175954064051
            ),
            (13, 14): InstructionProperties(
                duration=101 * chunk_size * dt, error=0.005231334872256355
            ),
        }
        self.target.add_instruction(RZXGate(np.pi / 2), rzx_90_props, name="rzx_90")
        rzx_45_props = {
            (0, 1): InstructionProperties(
                duration=52 * chunk_size * dt, error=0.030671121181161276 / 2
            ),
            (4, 1): InstructionProperties(
                duration=37 * chunk_size * dt, error=0.014041986073052737 / 2
            ),
            (4, 7): InstructionProperties(
                duration=40 * chunk_size * dt, error=0.0052040275323747 / 2
            ),
            (10, 7): InstructionProperties(
                duration=46 * chunk_size * dt, error=0.005625282141655502 / 2
            ),
            (10, 12): InstructionProperties(
                duration=45 * chunk_size * dt, error=0.005771827440726435 / 2
            ),
            (15, 12): InstructionProperties(
                duration=42 * chunk_size * dt, error=0.0050335609425562755 / 2
            ),
            (15, 18): InstructionProperties(
                duration=34 * chunk_size * dt, error=0.0051374141171115495 / 2
            ),
            (12, 13): InstructionProperties(
                duration=37 * chunk_size * dt, error=0.011361175954064051 / 2
            ),
            (13, 14): InstructionProperties(
                duration=52 * chunk_size * dt, error=0.005231334872256355 / 2
            ),
        }
        self.target.add_instruction(RZXGate(np.pi / 4), rzx_45_props, name="rzx_45")
        rzx_30_props = {
            (0, 1): InstructionProperties(
                duration=37 * chunk_size * dt, error=0.030671121181161276 / 3
            ),
            (4, 1): InstructionProperties(
                duration=24 * chunk_size * dt, error=0.014041986073052737 / 3
            ),
            (4, 7): InstructionProperties(
                duration=29 * chunk_size * dt, error=0.0052040275323747 / 3
            ),
            (10, 7): InstructionProperties(
                duration=32 * chunk_size * dt, error=0.005625282141655502 / 3
            ),
            (10, 12): InstructionProperties(
                duration=32 * chunk_size * dt, error=0.005771827440726435 / 3
            ),
            (15, 12): InstructionProperties(
                duration=29 * chunk_size * dt, error=0.0050335609425562755 / 3
            ),
            (15, 18): InstructionProperties(
                duration=26 * chunk_size * dt, error=0.0051374141171115495 / 3
            ),
            (12, 13): InstructionProperties(
                duration=24 * chunk_size * dt, error=0.011361175954064051 / 3
            ),
            (13, 14): InstructionProperties(
                duration=377 * chunk_size * dt, error=0.005231334872256355 / 3
            ),
        }
        self.target.add_instruction(RZXGate(np.pi / 6), rzx_30_props, name="rzx_30")
        reset_props = {(i,): InstructionProperties(duration=3676.4444444444443) for i in range(27)}
        self._target.add_instruction(Reset(), reset_props)

        meas_props = {
            (0,): InstructionProperties(duration=3.552e-06, error=0.02089999999999992),
            (1,): InstructionProperties(duration=3.552e-06, error=0.020199999999999996),
            (2,): InstructionProperties(duration=3.552e-06, error=0.014100000000000001),
            (3,): InstructionProperties(duration=3.552e-06, error=0.03710000000000002),
            (4,): InstructionProperties(duration=3.552e-06, error=0.015100000000000002),
            (5,): InstructionProperties(duration=3.552e-06, error=0.01869999999999994),
            (6,): InstructionProperties(duration=3.552e-06, error=0.013000000000000012),
            (7,): InstructionProperties(duration=3.552e-06, error=0.02059999999999995),
            (8,): InstructionProperties(duration=3.552e-06, error=0.06099999999999994),
            (9,): InstructionProperties(duration=3.552e-06, error=0.02950000000000008),
            (10,): InstructionProperties(duration=3.552e-06, error=0.040000000000000036),
            (11,): InstructionProperties(duration=3.552e-06, error=0.017299999999999982),
            (12,): InstructionProperties(duration=3.552e-06, error=0.04410000000000003),
            (13,): InstructionProperties(duration=3.552e-06, error=0.017199999999999993),
            (14,): InstructionProperties(duration=3.552e-06, error=0.10119999999999996),
            (15,): InstructionProperties(duration=3.552e-06, error=0.07840000000000003),
            (16,): InstructionProperties(duration=3.552e-06, error=0.014499999999999957),
            (17,): InstructionProperties(duration=3.552e-06, error=0.021299999999999986),
            (18,): InstructionProperties(duration=3.552e-06, error=0.022399999999999975),
            (19,): InstructionProperties(duration=3.552e-06, error=0.01859999999999995),
            (20,): InstructionProperties(duration=3.552e-06, error=0.02859999999999996),
            (21,): InstructionProperties(duration=3.552e-06, error=0.021600000000000064),
            (22,): InstructionProperties(duration=3.552e-06, error=0.030200000000000005),
            (23,): InstructionProperties(duration=3.552e-06, error=0.01970000000000005),
            (24,): InstructionProperties(duration=3.552e-06, error=0.03079999999999994),
            (25,): InstructionProperties(duration=3.552e-06, error=0.04400000000000004),
            (26,): InstructionProperties(duration=3.552e-06, error=0.026800000000000046),
        }
        self.target.add_instruction(Measure(), meas_props)
        self._qubit_properties = {
            0: QubitProperties(
                t1=0.00015987993124584417, t2=0.00016123516590787283, frequency=5073462814.921423
            ),
            1: QubitProperties(
                t1=0.00017271188343294773, t2=3.653713654834547e-05, frequency=4943844681.620448
            ),
            2: QubitProperties(
                t1=7.179635917914033e-05, t2=0.00012399765778639733, frequency=4668157502.363186
            ),
            3: QubitProperties(
                t1=0.0001124203171256432, t2=0.0001879954854434302, frequency=4887315883.214115
            ),
            4: QubitProperties(
                t1=9.568769051084652e-05, t2=6.9955557231525e-05, frequency=5016355075.77537
            ),
            5: QubitProperties(
                t1=9.361326963775646e-05, t2=0.00012561361411231962, frequency=4950539585.866738
            ),
            6: QubitProperties(
                t1=9.735672898365994e-05, t2=0.00012522003396944046, frequency=4970622491.726983
            ),
            7: QubitProperties(
                t1=0.00012117839009784141, t2=0.0001492370106539427, frequency=4889863864.167805
            ),
            8: QubitProperties(
                t1=8.394707006435891e-05, t2=5.5194256398727296e-05, frequency=4769852625.405966
            ),
            9: QubitProperties(
                t1=0.00012392229685657686, t2=5.97129502818714e-05, frequency=4948868138.885028
            ),
            10: QubitProperties(
                t1=0.00011193014813922708, t2=0.00014091085124119432, frequency=4966294754.357908
            ),
            11: QubitProperties(
                t1=0.000124426408667364, t2=9.561432905002298e-05, frequency=4664636564.282378
            ),
            12: QubitProperties(
                t1=0.00012469120424014884, t2=7.1792446286313e-05, frequency=4741461907.952719
            ),
            13: QubitProperties(
                t1=0.00010010942474357871, t2=9.260751861141544e-05, frequency=4879064835.799635
            ),
            14: QubitProperties(
                t1=0.00010793367069728063, t2=0.00020462601085738193, frequency=4774809501.962878
            ),
            15: QubitProperties(
                t1=0.00010814279470918582, t2=0.00014052616328020083, frequency=4860834948.367331
            ),
            16: QubitProperties(
                t1=9.889617874757627e-05, t2=0.00012160357011388956, frequency=4978318747.333388
            ),
            17: QubitProperties(
                t1=8.435212562619916e-05, t2=4.43587633824445e-05, frequency=5000300619.491221
            ),
            18: QubitProperties(
                t1=0.00011719166507869474, t2=5.461866556148401e-05, frequency=4772460318.985625
            ),
            19: QubitProperties(
                t1=0.00013321880066203932, t2=0.0001704632622810825, frequency=4807707035.998121
            ),
            20: QubitProperties(
                t1=9.14192211953385e-05, t2=0.00014298332288799443, frequency=5045028334.669125
            ),
            21: QubitProperties(
                t1=5.548103716494676e-05, t2=9.328101902519704e-05, frequency=4941029753.792485
            ),
            22: QubitProperties(
                t1=0.00017109481586484562, t2=0.00019209594920551097, frequency=4906801587.246266
            ),
            23: QubitProperties(
                t1=0.00010975552427765991, t2=0.00015616813868639905, frequency=4891601685.652732
            ),
            24: QubitProperties(
                t1=0.0001612962696960434, t2=6.940808472789023e-05, frequency=4664347869.784967
            ),
            25: QubitProperties(
                t1=0.00015414506978323392, t2=8.382170181880107e-05, frequency=4742061753.511209
            ),
            26: QubitProperties(
                t1=0.00011828557676958944, t2=0.00016963640893557827, frequency=4961661099.733828
            ),
        }

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, run_input, **options):
        raise NotImplementedError

    def qubit_properties(self, qubit):
        if isinstance(qubit, int):
            return self._qubit_properties[qubit]
        return [self._qubit_properties[i] for i in qubit]
