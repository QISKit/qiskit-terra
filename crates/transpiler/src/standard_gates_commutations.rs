// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use smallvec::smallvec;

use crate::commutation_checker::{CommutationLibrary, CommutationLibraryEntry};
use qiskit_circuit::Qubit;

static SIMPLE_COMMUTE: [([&str; 2], bool); 348] = [
    (["id", "id"], true),
    (["id", "sx"], true),
    (["id", "cx"], true),
    (["id", "c3sx"], true),
    (["id", "ccx"], true),
    (["id", "dcx"], true),
    (["id", "ch"], true),
    (["id", "cswap"], true),
    (["id", "csx"], true),
    (["id", "cy"], true),
    (["id", "cz"], true),
    (["id", "ccz"], true),
    (["id", "rccx"], true),
    (["id", "rcccx"], true),
    (["id", "ecr"], true),
    (["id", "sdg"], true),
    (["id", "cs"], true),
    (["id", "csdg"], true),
    (["id", "swap"], true),
    (["id", "iswap"], true),
    (["id", "sxdg"], true),
    (["id", "tdg"], true),
    (["id", "crx"], true),
    (["id", "cry"], true),
    (["id", "crz"], true),
    (["id", "rxx"], true),
    (["id", "ryy"], true),
    (["id", "rzz"], true),
    (["id", "rzx"], true),
    (["sx", "sx"], true),
    (["sx", "dcx"], false),
    (["sx", "ch"], false),
    (["sx", "cswap"], false),
    (["sx", "cy"], false),
    (["sx", "cz"], false),
    (["sx", "ccz"], false),
    (["sx", "rccx"], false),
    (["sx", "rcccx"], false),
    (["sx", "sdg"], false),
    (["sx", "cs"], false),
    (["sx", "csdg"], false),
    (["sx", "swap"], false),
    (["sx", "iswap"], false),
    (["sx", "sxdg"], true),
    (["sx", "tdg"], false),
    (["sx", "cry"], false),
    (["sx", "crz"], false),
    (["sx", "rxx"], true),
    (["sx", "ryy"], false),
    (["sx", "rzz"], false),
    (["x", "id"], true),
    (["x", "sx"], true),
    (["x", "x"], true),
    (["x", "dcx"], false),
    (["x", "ch"], false),
    (["x", "cswap"], false),
    (["x", "cy"], false),
    (["x", "cz"], false),
    (["x", "ccz"], false),
    (["x", "rccx"], false),
    (["x", "rcccx"], false),
    (["x", "sdg"], false),
    (["x", "cs"], false),
    (["x", "csdg"], false),
    (["x", "swap"], false),
    (["x", "iswap"], false),
    (["x", "sxdg"], true),
    (["x", "tdg"], false),
    (["x", "y"], false),
    (["x", "z"], false),
    (["x", "cry"], false),
    (["x", "crz"], false),
    (["x", "rxx"], true),
    (["x", "ryy"], false),
    (["x", "rzz"], false),
    (["cx", "dcx"], false),
    (["cx", "swap"], false),
    (["cx", "iswap"], false),
    (["cx", "ryy"], false),
    (["dcx", "c3sx"], false),
    (["dcx", "ccx"], false),
    (["dcx", "cswap"], false),
    (["dcx", "ccz"], false),
    (["dcx", "rccx"], false),
    (["dcx", "rcccx"], false),
    (["dcx", "ecr"], false),
    (["dcx", "csdg"], false),
    (["dcx", "swap"], false),
    (["dcx", "iswap"], false),
    (["dcx", "rxx"], false),
    (["dcx", "ryy"], false),
    (["dcx", "rzz"], false),
    (["dcx", "rzx"], false),
    (["ch", "dcx"], false),
    (["ch", "ecr"], false),
    (["ch", "swap"], false),
    (["ch", "iswap"], false),
    (["ch", "rxx"], false),
    (["ch", "ryy"], false),
    (["csx", "dcx"], false),
    (["csx", "swap"], false),
    (["csx", "iswap"], false),
    (["csx", "ryy"], false),
    (["cy", "dcx"], false),
    (["cy", "ecr"], false),
    (["cy", "swap"], false),
    (["cy", "iswap"], false),
    (["cy", "rxx"], false),
    (["cz", "dcx"], false),
    (["cz", "cz"], true),
    (["cz", "ccz"], true),
    (["cz", "ecr"], false),
    (["cz", "csdg"], true),
    (["cz", "crz"], true),
    (["cz", "rxx"], false),
    (["cz", "ryy"], false),
    (["cz", "rzz"], true),
    (["ccz", "ccz"], true),
    (["h", "id"], true),
    (["h", "sx"], false),
    (["h", "x"], false),
    (["h", "cx"], false),
    (["h", "c3sx"], false),
    (["h", "ccx"], false),
    (["h", "dcx"], false),
    (["h", "cswap"], false),
    (["h", "csx"], false),
    (["h", "cy"], false),
    (["h", "cz"], false),
    (["h", "ccz"], false),
    (["h", "h"], true),
    (["h", "rccx"], false),
    (["h", "rcccx"], false),
    (["h", "ecr"], false),
    (["h", "s"], false),
    (["h", "sdg"], false),
    (["h", "cs"], false),
    (["h", "csdg"], false),
    (["h", "swap"], false),
    (["h", "iswap"], false),
    (["h", "sxdg"], false),
    (["h", "t"], false),
    (["h", "tdg"], false),
    (["h", "y"], false),
    (["h", "z"], false),
    (["h", "crx"], false),
    (["h", "cry"], false),
    (["h", "crz"], false),
    (["h", "rxx"], false),
    (["h", "ryy"], false),
    (["h", "rzz"], false),
    (["h", "rzx"], false),
    (["ecr", "cswap"], false),
    (["ecr", "ccz"], false),
    (["ecr", "rccx"], false),
    (["ecr", "rcccx"], false),
    (["ecr", "csdg"], false),
    (["ecr", "swap"], false),
    (["ecr", "iswap"], false),
    (["ecr", "ryy"], false),
    (["ecr", "rzz"], false),
    (["s", "id"], true),
    (["s", "sx"], false),
    (["s", "x"], false),
    (["s", "dcx"], false),
    (["s", "cz"], true),
    (["s", "ccz"], true),
    (["s", "ecr"], false),
    (["s", "s"], true),
    (["s", "sdg"], true),
    (["s", "cs"], true),
    (["s", "csdg"], true),
    (["s", "swap"], false),
    (["s", "iswap"], false),
    (["s", "sxdg"], false),
    (["s", "t"], true),
    (["s", "tdg"], true),
    (["s", "y"], false),
    (["s", "z"], true),
    (["s", "crz"], true),
    (["s", "rxx"], false),
    (["s", "ryy"], false),
    (["s", "rzz"], true),
    (["sdg", "dcx"], false),
    (["sdg", "cz"], true),
    (["sdg", "ccz"], true),
    (["sdg", "ecr"], false),
    (["sdg", "sdg"], true),
    (["sdg", "cs"], true),
    (["sdg", "csdg"], true),
    (["sdg", "swap"], false),
    (["sdg", "iswap"], false),
    (["sdg", "sxdg"], false),
    (["sdg", "tdg"], true),
    (["sdg", "crz"], true),
    (["sdg", "rxx"], false),
    (["sdg", "ryy"], false),
    (["sdg", "rzz"], true),
    (["cs", "dcx"], false),
    (["cs", "cz"], true),
    (["cs", "ccz"], true),
    (["cs", "ecr"], false),
    (["cs", "cs"], true),
    (["cs", "csdg"], true),
    (["cs", "crz"], true),
    (["cs", "rxx"], false),
    (["cs", "ryy"], false),
    (["cs", "rzz"], true),
    (["csdg", "ccz"], true),
    (["csdg", "csdg"], true),
    (["swap", "rccx"], false),
    (["iswap", "rccx"], false),
    (["sxdg", "dcx"], false),
    (["sxdg", "ch"], false),
    (["sxdg", "cswap"], false),
    (["sxdg", "cy"], false),
    (["sxdg", "cz"], false),
    (["sxdg", "ccz"], false),
    (["sxdg", "rccx"], false),
    (["sxdg", "rcccx"], false),
    (["sxdg", "cs"], false),
    (["sxdg", "csdg"], false),
    (["sxdg", "swap"], false),
    (["sxdg", "iswap"], false),
    (["sxdg", "sxdg"], true),
    (["sxdg", "cry"], false),
    (["sxdg", "crz"], false),
    (["sxdg", "rxx"], true),
    (["sxdg", "ryy"], false),
    (["sxdg", "rzz"], false),
    (["t", "id"], true),
    (["t", "sx"], false),
    (["t", "x"], false),
    (["t", "dcx"], false),
    (["t", "cz"], true),
    (["t", "ccz"], true),
    (["t", "ecr"], false),
    (["t", "sdg"], true),
    (["t", "cs"], true),
    (["t", "csdg"], true),
    (["t", "swap"], false),
    (["t", "iswap"], false),
    (["t", "sxdg"], false),
    (["t", "t"], true),
    (["t", "tdg"], true),
    (["t", "y"], false),
    (["t", "z"], true),
    (["t", "crz"], true),
    (["t", "rxx"], false),
    (["t", "ryy"], false),
    (["t", "rzz"], true),
    (["tdg", "dcx"], false),
    (["tdg", "cz"], true),
    (["tdg", "ccz"], true),
    (["tdg", "ecr"], false),
    (["tdg", "cs"], true),
    (["tdg", "csdg"], true),
    (["tdg", "swap"], false),
    (["tdg", "iswap"], false),
    (["tdg", "sxdg"], false),
    (["tdg", "tdg"], true),
    (["tdg", "crz"], true),
    (["tdg", "rxx"], false),
    (["tdg", "ryy"], false),
    (["tdg", "rzz"], true),
    (["y", "id"], true),
    (["y", "sx"], false),
    (["y", "cx"], false),
    (["y", "c3sx"], false),
    (["y", "ccx"], false),
    (["y", "dcx"], false),
    (["y", "ch"], false),
    (["y", "cswap"], false),
    (["y", "csx"], false),
    (["y", "cz"], false),
    (["y", "ccz"], false),
    (["y", "rccx"], false),
    (["y", "rcccx"], false),
    (["y", "ecr"], false),
    (["y", "sdg"], false),
    (["y", "cs"], false),
    (["y", "csdg"], false),
    (["y", "swap"], false),
    (["y", "iswap"], false),
    (["y", "sxdg"], false),
    (["y", "tdg"], false),
    (["y", "y"], true),
    (["y", "z"], false),
    (["y", "crx"], false),
    (["y", "crz"], false),
    (["y", "rxx"], false),
    (["y", "ryy"], true),
    (["y", "rzz"], false),
    (["y", "rzx"], false),
    (["z", "id"], true),
    (["z", "sx"], false),
    (["z", "dcx"], false),
    (["z", "cz"], true),
    (["z", "ccz"], true),
    (["z", "ecr"], false),
    (["z", "sdg"], true),
    (["z", "cs"], true),
    (["z", "csdg"], true),
    (["z", "swap"], false),
    (["z", "iswap"], false),
    (["z", "sxdg"], false),
    (["z", "tdg"], true),
    (["z", "z"], true),
    (["z", "crz"], true),
    (["z", "rxx"], false),
    (["z", "ryy"], false),
    (["z", "rzz"], true),
    (["crx", "dcx"], false),
    (["crx", "swap"], false),
    (["crx", "iswap"], false),
    (["crx", "ryy"], false),
    (["cry", "dcx"], false),
    (["cry", "ecr"], false),
    (["cry", "swap"], false),
    (["cry", "iswap"], false),
    (["cry", "rxx"], false),
    (["crz", "dcx"], false),
    (["crz", "ccz"], true),
    (["crz", "ecr"], false),
    (["crz", "csdg"], true),
    (["crz", "swap"], false),
    (["crz", "iswap"], false),
    (["crz", "crz"], true),
    (["crz", "rxx"], false),
    (["crz", "ryy"], false),
    (["crz", "rzz"], true),
    (["rxx", "ccz"], false),
    (["rxx", "rccx"], false),
    (["rxx", "rcccx"], false),
    (["rxx", "csdg"], false),
    (["rxx", "rxx"], true),
    (["ryy", "c3sx"], false),
    (["ryy", "ccx"], false),
    (["ryy", "ccz"], false),
    (["ryy", "rccx"], false),
    (["ryy", "rcccx"], false),
    (["ryy", "csdg"], false),
    (["ryy", "ryy"], true),
    (["rzz", "ccz"], true),
    (["rzz", "csdg"], true),
    (["rzz", "rzz"], true),
    (["rzx", "swap"], false),
    (["rzx", "iswap"], false),
];

pub fn get_commutation_library() -> CommutationLibrary {
    let mut commutation_library = CommutationLibrary::with_capacity(630);
    for (key, value) in SIMPLE_COMMUTE {
        commutation_library.add_entry(key, CommutationLibraryEntry::Commutes(value));
    }
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["x", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "cz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "crz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "rxx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), None, None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), None, None, None,], false),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), None, Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(0)), None, None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), None, Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(1)), None, None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (smallvec![None, Some(Qubit(2)), None, Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(2)), None, None,], true),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, Some(Qubit(3)), None, None,], false),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![None, None, Some(Qubit(0)), None,], true),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![None, None, Some(Qubit(1)), None,], true),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![None, None, Some(Qubit(2)), None,], true),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["c3sx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), None, None, None,], false),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), None, None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), None, None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(2)), None, None,], true),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, Some(Qubit(3)), None, None,], false),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(0)), None,], true),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(1)), None,], true),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(2)), None,], true),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["c3sx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["dcx", "dcx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cs"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "crz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ch", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cswap", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cswap", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cswap", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "rxx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "cz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "crz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "ryy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cy", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cz", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], true),
        (smallvec![None, None, Some(Qubit(1)),], true),
        (smallvec![None, None, Some(Qubit(2)),], true),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccz", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None, None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], true),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccz", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], true),
        (smallvec![None, None, Some(Qubit(1)),], true),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccz", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], true),
        (smallvec![None, None, Some(Qubit(1)),], true),
        (smallvec![None, None, Some(Qubit(2)),], true),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ccz", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["h", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rccx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rccx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rccx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None, None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None, None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)), None,], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None, None,], true),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)), None,], false),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), None, Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None, None,], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(0)), None,], true),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![None, Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![None, Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rccx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(0)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(0)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(0)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)), None, None,], true),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(1)), None, Some(Qubit(2)), None,], true),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(1)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(1)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(3)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(3)),
            ],
            true,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)), None,],
            true,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(3)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)), None, None,], true),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(2)),
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), Some(Qubit(3)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(0)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![Some(Qubit(2)), None, Some(Qubit(1)), None,], true),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(2)), None, None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![Some(Qubit(2)), None, None, None,], true),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(1)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(0)),
                Some(Qubit(2)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(0)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(0)),
                Some(Qubit(2)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(1)),
                Some(Qubit(2)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(1)), None, None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(0)),
                Some(Qubit(1)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![
                Some(Qubit(3)),
                Some(Qubit(2)),
                Some(Qubit(1)),
                Some(Qubit(0)),
            ],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), Some(Qubit(2)), None, None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![Some(Qubit(3)), None, None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![Some(Qubit(3)), None, None, None,], false),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(0)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(0)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(0)), None, None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(2)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(1)), Some(Qubit(2)), None,], true),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(1)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(1)), None, None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(0)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(0)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(1)), Some(Qubit(3)),],
            true,
        ),
        (smallvec![None, Some(Qubit(2)), Some(Qubit(1)), None,], true),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), Some(Qubit(3)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(2)), None, Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, Some(Qubit(2)), None, None,], true),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(0)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(1)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), Some(Qubit(2)), None,],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, Some(Qubit(3)), None, Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, Some(Qubit(3)), None, None,], false),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(0)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(0)), None,], true),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(2)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(1)), Some(Qubit(3)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(1)), None,], true),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(2)), Some(Qubit(1)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![None, None, Some(Qubit(2)), None,], true),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(0)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(1)),],
            false,
        ),
        (
            smallvec![None, None, Some(Qubit(3)), Some(Qubit(2)),],
            false,
        ),
        (smallvec![None, None, Some(Qubit(3)), None,], false),
        (smallvec![None, None, None, Some(Qubit(0)),], false),
        (smallvec![None, None, None, Some(Qubit(1)),], false),
        (smallvec![None, None, None, Some(Qubit(2)),], false),
        (smallvec![None, None, None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rcccx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ecr", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ecr", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ecr", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ecr", "rxx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ecr", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["s", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sdg", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cs", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["csdg", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["swap", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["iswap", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["sxdg", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["t", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["tdg", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["y", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["y", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "cx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "ch"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "cy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["z", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "ecr"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "crx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "crz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "rxx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "cry"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "crz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "ryy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["cry", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "csx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["crz", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "ryy"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rxx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ryy", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ryy", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ryy", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ryy", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["ryy", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], true),
        (smallvec![None, Some(Qubit(1)),], true),
        (smallvec![None, Some(Qubit(2)),], true),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "swap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzz", "iswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], true),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "c3sx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], true),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "ccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "cswap"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "ccz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "rccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(0)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(1)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![Some(Qubit(2)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(2)), Some(Qubit(3)),], false),
        (smallvec![Some(Qubit(2)), None,], true),
        (smallvec![Some(Qubit(3)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(3)), Some(Qubit(2)),], false),
        (smallvec![Some(Qubit(3)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
        (smallvec![None, Some(Qubit(2)),], false),
        (smallvec![None, Some(Qubit(3)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "rcccx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "csdg"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], false),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], false),
        (smallvec![Some(Qubit(1)), None,], true),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], false),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "rzz"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    let commutation_map = [
        (smallvec![Some(Qubit(0)), Some(Qubit(1)),], true),
        (smallvec![Some(Qubit(0)), None,], true),
        (smallvec![Some(Qubit(1)), Some(Qubit(0)),], true),
        (smallvec![Some(Qubit(1)), None,], false),
        (smallvec![None, Some(Qubit(0)),], false),
        (smallvec![None, Some(Qubit(1)),], true),
    ]
    .into_iter()
    .collect();
    commutation_library.add_entry(
        ["rzx", "rzx"],
        CommutationLibraryEntry::QubitMapping(commutation_map),
    );
    commutation_library
}
