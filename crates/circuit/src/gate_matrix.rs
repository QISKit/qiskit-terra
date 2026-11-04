// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

use qiskit_util::alias::{GateArray0Q, GateArray1Q, GateArray2Q, GateArray3Q, GateArray4Q};
use qiskit_util::complex::{C_M_ONE, C_ONE, C_ZERO, IM, M_IM, c64};

pub static ONE_QUBIT_IDENTITY: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, C_ONE]];
pub static TWO_QUBIT_IDENTITY: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];

// Utility for generating static matrices for controlled gates with "n" control qubits.
// Assumptions:
// 1. the reference "gate-matrix" is a single-qubit gate matrix (2x2)
// 2. the first "n" qubits are controls and the last qubit is the target
macro_rules! make_n_controlled_gate {
    ($gate_matrix:expr, $n_control_qubits:expr) => {{
        const DIM: usize = 2_usize.pow($n_control_qubits as u32 + 1_u32);
        // DIM x DIM matrix of all zeros
        let mut matrix: [[Complex64; DIM]; DIM] = [[C_ZERO; DIM]; DIM];
        // DIM x DIM diagonal matrix
        {
            let mut i = 0;
            while i < DIM {
                matrix[i][i] = C_ONE;
                i += 1;
            }
        }
        // Insert elements of gate_matrix in columns DIM/2-1 and DIM-1
        matrix[DIM / 2 - 1][DIM / 2 - 1] = $gate_matrix[0][0];
        matrix[DIM - 1][DIM - 1] = $gate_matrix[1][1];
        matrix[DIM / 2 - 1][DIM - 1] = $gate_matrix[0][1];
        matrix[DIM - 1][DIM / 2 - 1] = $gate_matrix[1][0];
        matrix
    }};
}

pub static H_GATE: GateArray1Q = [
    [c64(FRAC_1_SQRT_2, 0.), c64(FRAC_1_SQRT_2, 0.)],
    [c64(FRAC_1_SQRT_2, 0.), c64(-FRAC_1_SQRT_2, 0.)],
];

pub static X_GATE: GateArray1Q = [[C_ZERO, C_ONE], [C_ONE, C_ZERO]];

pub static Z_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, C_M_ONE]];

pub static Y_GATE: GateArray1Q = [[C_ZERO, M_IM], [IM, C_ZERO]];

pub static S_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, IM]];

pub static SDG_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, M_IM]];

pub static SX_GATE: GateArray1Q = [
    [c64(0.5, 0.5), c64(0.5, -0.5)],
    [c64(0.5, -0.5), c64(0.5, 0.5)],
];

pub static SXDG_GATE: GateArray1Q = [
    [c64(0.5, -0.5), c64(0.5, 0.5)],
    [c64(0.5, 0.5), c64(0.5, -0.5)],
];

pub static T_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, c64(FRAC_1_SQRT_2, FRAC_1_SQRT_2)]];

pub static TDG_GATE: GateArray1Q = [
    [C_ONE, C_ZERO],
    [C_ZERO, c64(FRAC_1_SQRT_2, -FRAC_1_SQRT_2)],
];

pub static CH_GATE: GateArray2Q = make_n_controlled_gate!(H_GATE, 1);

pub static CX_GATE: GateArray2Q = make_n_controlled_gate!(X_GATE, 1);

pub static CY_GATE: GateArray2Q = make_n_controlled_gate!(Y_GATE, 1);

pub static CZ_GATE: GateArray2Q = make_n_controlled_gate!(Z_GATE, 1);

pub static DCX_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
];

pub static ECR_GATE: GateArray2Q = [
    [
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
    ],
    [
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
];

pub static SWAP_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];

pub static ISWAP_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, IM, C_ZERO],
    [C_ZERO, IM, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];

pub static CS_GATE: GateArray2Q = make_n_controlled_gate!(S_GATE, 1);

pub static CSDG_GATE: GateArray2Q = make_n_controlled_gate!(SDG_GATE, 1);

pub static CSX_GATE: GateArray2Q = make_n_controlled_gate!(SX_GATE, 1);

pub static CCX_GATE: GateArray3Q = make_n_controlled_gate!(X_GATE, 2);

pub static CCZ_GATE: GateArray3Q = make_n_controlled_gate!(Z_GATE, 2);

pub static CSWAP_GATE: GateArray3Q = [
    [
        C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE,
    ],
];

pub static RCCX_GATE: GateArray3Q = [
    [
        C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, M_IM],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_M_ONE, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO,
    ],
    [C_ZERO, C_ZERO, C_ZERO, IM, C_ZERO, C_ZERO, C_ZERO, C_ZERO],
];

pub static C3X_GATE: GateArray4Q = make_n_controlled_gate!(X_GATE, 3);

pub static C3SX_GATE: GateArray4Q = make_n_controlled_gate!(SX_GATE, 3);

pub static RC3X_GATE: GateArray4Q = [
    [
        C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, IM, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        M_IM, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_M_ONE, C_ZERO, C_ZERO, C_ZERO,
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
];

#[inline]
pub fn global_phase_gate(theta: f64) -> GateArray0Q {
    [[c64(0., theta).exp()]]
}

#[inline]
pub fn phase_gate(lam: f64) -> GateArray1Q {
    [[C_ONE, C_ZERO], [C_ZERO, c64(0., lam).exp()]]
}

#[inline]
pub fn r_gate(theta: f64, phi: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cost = c64(half_theta.cos(), 0.);
    let sint = half_theta.sin();
    let cosphi = phi.cos();
    let sinphi = phi.sin();
    [
        [cost, c64(-sint * sinphi, -sint * cosphi)],
        [c64(sint * sinphi, -sint * cosphi), cost],
    ]
}

#[inline]
pub fn rx_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

#[inline]
pub fn ry_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    [[cos, -sin], [sin, cos]]
}

#[inline]
pub fn rz_gate(theta: f64) -> GateArray1Q {
    let ilam2 = c64(0., 0.5 * theta);
    [[(-ilam2).exp(), C_ZERO], [C_ZERO, ilam2.exp()]]
}

#[inline]
pub fn u_gate(theta: f64, phi: f64, lam: f64) -> GateArray1Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), (-c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn u1_gate(lam: f64) -> GateArray1Q {
    [[C_ONE, C_ZERO], [C_ZERO, c64(0., lam).exp()]]
}

#[inline]
pub fn u2_gate(phi: f64, lam: f64) -> GateArray1Q {
    [
        [
            c64(FRAC_1_SQRT_2, 0.),
            (-c64(0., lam).exp()) * FRAC_1_SQRT_2,
        ],
        [
            c64(0., phi).exp() * FRAC_1_SQRT_2,
            c64(0., phi + lam).exp() * FRAC_1_SQRT_2,
        ],
    ]
}

#[inline]
pub fn u3_gate(theta: f64, phi: f64, lam: f64) -> GateArray1Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), -(c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn cp_gate(lam: f64) -> GateArray2Q {
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [C_ZERO, C_ONE, C_ZERO, C_ZERO],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [C_ZERO, C_ZERO, C_ZERO, c64(0., lam).exp()],
    ]
}

#[inline]
pub fn crx_gate(theta: f64) -> GateArray2Q {
    let gate_matrix = rx_gate(theta);
    make_n_controlled_gate!(gate_matrix, 1)
}

#[inline]
pub fn cry_gate(theta: f64) -> GateArray2Q {
    let gate_matrix = ry_gate(theta);
    make_n_controlled_gate!(gate_matrix, 1)
}

#[inline]
pub fn crz_gate(theta: f64) -> GateArray2Q {
    let gate_matrix = rz_gate(theta);
    make_n_controlled_gate!(gate_matrix, 1)
}

#[inline]
pub fn cu_gate(theta: f64, phi: f64, lam: f64, gamma: f64) -> GateArray2Q {
    let cos_theta = (theta / 2.).cos();
    let sin_theta = (theta / 2.).sin();
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [
            C_ZERO,
            c64(0., gamma).exp() * cos_theta,
            C_ZERO,
            c64(0., gamma + lam).exp() * (-1.) * sin_theta,
        ],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [
            C_ZERO,
            c64(0., gamma + phi).exp() * sin_theta,
            C_ZERO,
            c64(0., gamma + phi + lam).exp() * cos_theta,
        ],
    ]
}

#[inline]
pub fn cu1_gate(lam: f64) -> GateArray2Q {
    let gate_matrix = u1_gate(lam);
    make_n_controlled_gate!(gate_matrix, 1)
}

#[inline]
pub fn cu3_gate(theta: f64, phi: f64, lam: f64) -> GateArray2Q {
    let gate_matrix = u3_gate(theta, phi, lam);
    make_n_controlled_gate!(gate_matrix, 1)
}

#[inline]
pub fn rxx_gate(theta: f64) -> GateArray2Q {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csinm = c64(0., -sint);

    [
        [ccos, C_ZERO, C_ZERO, csinm],
        [C_ZERO, ccos, csinm, C_ZERO],
        [C_ZERO, csinm, ccos, C_ZERO],
        [csinm, C_ZERO, C_ZERO, ccos],
    ]
}

#[inline]
pub fn ryy_gate(theta: f64) -> GateArray2Q {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csin = c64(0., sint);

    [
        [ccos, C_ZERO, C_ZERO, csin],
        [C_ZERO, ccos, -csin, C_ZERO],
        [C_ZERO, -csin, ccos, C_ZERO],
        [csin, C_ZERO, C_ZERO, ccos],
    ]
}

#[inline]
pub fn rzz_gate(theta: f64) -> GateArray2Q {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let exp_it2 = c64(cost, sint);
    let exp_mit2 = c64(cost, -sint);

    [
        [exp_mit2, C_ZERO, C_ZERO, C_ZERO],
        [C_ZERO, exp_it2, C_ZERO, C_ZERO],
        [C_ZERO, C_ZERO, exp_it2, C_ZERO],
        [C_ZERO, C_ZERO, C_ZERO, exp_mit2],
    ]
}

#[inline]
pub fn rzx_gate(theta: f64) -> GateArray2Q {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csin = c64(0., sint);

    [
        [ccos, C_ZERO, -csin, C_ZERO],
        [C_ZERO, ccos, C_ZERO, csin],
        [-csin, C_ZERO, ccos, C_ZERO],
        [C_ZERO, csin, C_ZERO, ccos],
    ]
}

#[inline]
pub fn xx_minus_yy_gate(theta: f64, beta: f64) -> GateArray2Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [
            c64(cos, 0.),
            C_ZERO,
            C_ZERO,
            c64(0., -sin) * c64(0., -beta).exp(),
        ],
        [C_ZERO, C_ONE, C_ZERO, C_ZERO],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [
            c64(0., -sin) * c64(0., beta).exp(),
            C_ZERO,
            C_ZERO,
            c64(cos, 0.),
        ],
    ]
}

#[inline]
pub fn xx_plus_yy_gate(theta: f64, beta: f64) -> GateArray2Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [
            C_ZERO,
            c64(cos, 0.),
            c64(0., -sin) * c64(0., -beta).exp(),
            C_ZERO,
        ],
        [
            C_ZERO,
            c64(0., -sin) * c64(0., beta).exp(),
            c64(cos, 0.),
            C_ZERO,
        ],
        [C_ZERO, C_ZERO, C_ZERO, C_ONE],
    ]
}
