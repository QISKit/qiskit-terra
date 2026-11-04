// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

const fn generate_lookup_table() -> [&'static str; 103] {
    let mut lookup = [""; 103];
    lookup[b'0' as usize] = "0000";
    lookup[b'1' as usize] = "0001";
    lookup[b'2' as usize] = "0010";
    lookup[b'3' as usize] = "0011";
    lookup[b'4' as usize] = "0100";
    lookup[b'5' as usize] = "0101";
    lookup[b'6' as usize] = "0110";
    lookup[b'7' as usize] = "0111";
    lookup[b'8' as usize] = "1000";
    lookup[b'9' as usize] = "1001";
    lookup[b'A' as usize] = "1010";
    lookup[b'B' as usize] = "1011";
    lookup[b'C' as usize] = "1100";
    lookup[b'D' as usize] = "1101";
    lookup[b'E' as usize] = "1110";
    lookup[b'F' as usize] = "1111";
    lookup[b'a' as usize] = "1010";
    lookup[b'b' as usize] = "1011";
    lookup[b'c' as usize] = "1100";
    lookup[b'd' as usize] = "1101";
    lookup[b'e' as usize] = "1110";
    lookup[b'f' as usize] = "1111";
    lookup
}

static HEX_TO_BIN_LUT: [&str; 103] = generate_lookup_table();

#[inline]
pub fn hex_to_bin(hex: &str) -> String {
    hex[2..]
        .chars()
        .map(|c| HEX_TO_BIN_LUT[c as usize])
        .collect()
}
