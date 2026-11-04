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

"""Post-processing of raw result."""

import numpy as np

from qiskit.exceptions import QiskitError


def _hex_to_bin(hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]


def _bin_to_hex(bitstring):
    """Convert bitstring readouts (memory) to hexadecimal readouts."""
    return hex(int(bitstring, 2))


def _pad_zeros(bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), f"0{memory_slots}b")


def _separate_bitstring(bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for _, size in reversed(creg_sizes):
        substrings.append(bitstring[running_index : running_index + size])
        running_index += size
    return " ".join(substrings)


def format_counts_memory(shot_memory, header=None):
    """
    Format a single bitstring (memory) from a single shot experiment.

    - The hexadecimals are expanded to bitstrings

    - Spaces are inserted at register divisions.

    Args:
        shot_memory (str): result of a single experiment.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing. creg_sizes
            are a nested list where the inner element is a list
            of creg name, creg size pairs. memory_slots is an integer
            specifying the number of total memory_slots in the experiment.

    Returns:
        str: a formatted memory
    """
    if shot_memory.startswith("0x"):
        shot_memory = _hex_to_bin(shot_memory)
    if header:
        creg_sizes = header.get("creg_sizes", None)
        memory_slots = header.get("memory_slots", None)
        if memory_slots:
            shot_memory = _pad_zeros(shot_memory, memory_slots)
        if creg_sizes and memory_slots:
            shot_memory = _separate_bitstring(shot_memory, creg_sizes)
    return shot_memory


def _list_to_complex_array(complex_list):
    """Convert nested list of shape (..., 2) to complex numpy array with shape (...)

    Args:
        complex_list (list): List to convert.

    Returns:
        np.ndarray: Complex numpy array

    Raises:
        QiskitError: If inner most array of input nested list is not of length 2.
    """
    arr = np.asarray(complex_list, dtype=np.complex128)
    if not arr.shape[-1] == 2:
        raise QiskitError("Inner most nested list is not of length 2.")

    return arr[..., 0] + 1j * arr[..., 1]


def format_level_0_memory(memory):
    """Format an experiment result memory object for measurement level 0.

    Args:
        memory (list): Memory from experiment with `meas_level==0`. `avg` or
            `single` will be inferred from shape of result memory.

    Returns:
        np.ndarray: Measurement level 0 complex numpy array

    Raises:
        QiskitError: If the returned numpy array does not have 2 (avg) or 3 (single)
            indices.
    """
    formatted_memory = _list_to_complex_array(memory)
    # infer meas_return from shape of returned data.
    if not 2 <= len(formatted_memory.shape) <= 3:
        raise QiskitError("Level zero memory is not of correct shape.")
    return formatted_memory


def format_level_1_memory(memory):
    """Format an experiment result memory object for measurement level 1.

    Args:
        memory (list): Memory from experiment with `meas_level==1`. `avg` or
            `single` will be inferred from shape of result memory.

    Returns:
        np.ndarray: Measurement level 1 complex numpy array

    Raises:
        QiskitError: If the returned numpy array does not have 1 (avg) or 2 (single)
            indices.
    """
    formatted_memory = _list_to_complex_array(memory)
    # infer meas_return from shape of returned data.
    if not 1 <= len(formatted_memory.shape) <= 2:
        raise QiskitError("Level one memory is not of correct shape.")
    return formatted_memory


def format_level_2_memory(memory, header=None):
    """Format an experiment result memory object for measurement level 2.

    Args:
        memory (list): Memory from experiment with `meas_level==2` and `memory==True`.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        list[str]: List of bitstrings
    """
    memory_list = []
    for shot_memory in memory:
        memory_list.append(format_counts_memory(shot_memory, header))
    return memory_list


def format_counts(counts, header=None):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.

    Args:
        counts (dict): counts histogram of multiple shots
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted counts
    """
    counts_dict = {}
    for key, val in counts.items():
        key = format_counts_memory(key, header)
        counts_dict[key] = val
    return counts_dict


def format_statevector(vec, decimals=None):
    """Format statevector coming from the backend to present to the Qiskit user.

    Args:
        vec (list): a list of [re, im] complex numbers.
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[complex]: a list of python complex numbers.
    """

    from qiskit.quantum_info.states.statevector import Statevector

    if isinstance(vec, Statevector):
        if decimals:
            return Statevector(np.around(vec.data, decimals=decimals), dims=vec.dims())
        return vec
    if isinstance(vec, np.ndarray):
        if decimals:
            return np.around(vec, decimals=decimals)
        return vec
    num_basis = len(vec)
    if vec and isinstance(vec[0], complex):
        vec_complex = np.array(vec, dtype=complex)
    else:
        vec_complex = np.zeros(num_basis, dtype=complex)
        for i in range(num_basis):
            vec_complex[i] = vec[i][0] + 1j * vec[i][1]

    if decimals:
        vec_complex = np.around(vec_complex, decimals=decimals)
    return vec_complex


def format_unitary(mat, decimals=None):
    """Format unitary coming from the backend to present to the Qiskit user.

    Args:
        mat (list[list]): a list of list of [re, im] complex numbers
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[list[complex]]: a matrix of complex numbers
    """

    from qiskit.quantum_info.operators.operator import Operator

    if isinstance(mat, Operator):
        if decimals:
            return Operator(
                np.around(mat.data, decimals=decimals),
                input_dims=mat.input_dims(),
                output_dims=mat.output_dims(),
            )
        return mat
    if isinstance(mat, np.ndarray):
        if decimals:
            return np.around(mat, decimals=decimals)
        return mat
    num_basis = len(mat)
    mat_complex = np.zeros((num_basis, num_basis), dtype=complex)
    for i, vec in enumerate(mat):
        mat_complex[i] = format_statevector(vec, decimals)
    return mat_complex
