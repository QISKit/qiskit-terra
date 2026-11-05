# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The original implementation of Qiskit's `parallel_map` in our commit c9c4ed52 was substantially
# derived from QuTiP's (https://github.com/qutip/qutip) in `qutip/parallel.py` at their commit
# f22d3cb7.  It has subsequently been significantly rewritten.
#
# The original implementation was used under these licence terms:
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

"""
Routines for running Python functions in parallel using process pools
from the multiprocessing library.
"""

from __future__ import annotations

import contextlib
import functools
import multiprocessing
import os
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor

from qiskit import user_config


CONFIG = user_config.get_config()


def _task_wrapper(param):
    (task, value, task_args, task_kwargs) = param
    return task(value, *task_args, **task_kwargs)


def _physical_cpus_assuming_twofold_smt():
    if (sched_getaffinity := getattr(os, "sched_getaffinity", None)) is not None:
        # It is callable, just pylint doesn't recognise it as `os.sched_getaffinity` because of the
        # `getattr`.

        num_cpus = len(sched_getaffinity(0))
    else:
        num_cpus = os.cpu_count() or 1
    return (num_cpus // 2) or 1


def _parallel_default():
    # We default to False on all platforms, unless parallelism is explicitly requested of fork-based implementations.
    return multiprocessing.get_start_method(allow_none=True) in ("fork", "forkserver")


@functools.cache
def default_num_processes() -> int:
    """Get the number of processes that a multiprocessing parallel call will use by default.

    Such functions typically also accept a ``num_processes`` keyword argument that will supersede
    the value returned from this function.

    In order of priority (highest to lowest), the return value will be:

    1. The ``QISKIT_NUM_PROCS`` environment variable, if set.
    2. The ``num_processes`` key of the Qiskit user configuration file, if set.
    3. Half of the logical CPUs available to this process, if this can be determined.  This is a
       proxy for the number of physical CPUs, assuming two-fold simultaneous multithreading (SMT);
       empirically, multiprocessing performance of Qiskit seems to be worse when attempting to use
       SMT cores.
    4. 1, if all else fails.

    If a user-configured value is set to a number less than 1, it is treated as if it were 1.
    """
    # Ignore both `None` (unset) and explicit set to empty string.
    if env_num_processes := os.getenv("QISKIT_NUM_PROCS"):
        try:
            env_num_processes = int(env_num_processes)
        except ValueError:
            # Invalid: fall back to other methods.
            warnings.warn(
                "failed to interpret environment 'QISKIT_NUM_PROCS' as a number:"
                f" '{env_num_processes}'"
            )
        else:
            return env_num_processes if env_num_processes > 0 else 1
    if (user_num_processes := CONFIG.get("num_processes", None)) is not None:
        return user_num_processes if user_num_processes > 0 else 1
    return _physical_cpus_assuming_twofold_smt()


def local_hardware_info():
    """Basic hardware information about the local machine.

    Attempts to estimate the number of physical CPUs in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """
    return {
        "python_compiler": platform.python_compiler(),
        "python_build": ", ".join(platform.python_build()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpus": _physical_cpus_assuming_twofold_smt(),
    }


def is_main_process() -> bool:
    """Checks whether the current process is the main one.

    Since Python 3.8, this is identical to the standard Python way of calculating this::

        >>> import multiprocessing
        >>> multiprocessing.parent_process() is None

    This function is left for backwards compatibility, but there is little reason not to use the
    built-in tooling of Python.
    """
    return multiprocessing.parent_process() is None


_PARALLEL_OVERRIDE = None
_PARALLEL_IGNORE_USER_SETTINGS = False
_IN_PARALLEL_ALLOW_PARALLELISM = "FALSE"
_IN_PARALLEL_FORBID_PARALLELISM = "TRUE"


@functools.cache
def should_run_in_parallel(num_processes: int | None = None) -> bool:
    """Decide whether a multiprocessing function should spawn subprocesses for parallelization.

    In particular, this is how :func:`parallel_map` decides whether to use multiprocessing or not.
    The ``num_processes`` argument alone does not enforce parallelism; by default, Qiskit disables
    process-based parallelism by default. It can enabled by explicitly setting a ``fork``-like
    :mod:`multiprocessing` start method, setting the ``QISKIT_PARALLEL`` environment variable to
    ``"TRUE"``, or setting ``parallel = true`` in your user settings file.

    This function includes two context managers that can be used to temporarily modify the return
    value of this function:

    .. autofunction:: qiskit.utils::should_run_in_parallel.override
    .. autofunction:: qiskit.utils::should_run_in_parallel.ignore_user_settings

    Args:
        num_processes: the maximum number of processes requested for use (``None`` implies the
            default).

    Examples:
        Temporarily override the configured settings to disable parallelism::

            >>> with should_run_in_parallel.override(True):
            ...     assert should_run_in_parallel(8)
            >>> with should_run_in_parallel.override(False):
            ...     assert not should_run_in_parallel(8)
    """
    # It's a configuration function with many simple choices - it'd be less clean to return late.

    num_processes = default_num_processes() if num_processes is None else num_processes
    if num_processes < 2:
        # There's no resources to parallelise over.
        return False
    if (
        os.getenv("QISKIT_IN_PARALLEL", _IN_PARALLEL_ALLOW_PARALLELISM)
        != _IN_PARALLEL_ALLOW_PARALLELISM
    ):
        # This isn't a user-set variable; we set this to talk to our own child processes.
        return False
    if _PARALLEL_OVERRIDE is not None:
        return _PARALLEL_OVERRIDE
    if _PARALLEL_IGNORE_USER_SETTINGS:
        return _parallel_default()
    if (env_qiskit_parallel := os.getenv("QISKIT_PARALLEL")) is not None:
        return env_qiskit_parallel.lower() == "true"
    if (user_qiskit_parallel := CONFIG.get("parallel_enabled", None)) is not None:
        return user_qiskit_parallel
    # Otherwise, fallback to the default.
    return _parallel_default()


@contextlib.contextmanager
def _parallel_ignore_user_settings():
    """A context manager within which :func:`should_run_in_parallel` will ignore environmental
    configuration variables.

    In particular, the ``QISKIT_PARALLEL`` environment variable and the user-configuration file are
    ignored within this context."""
    # The way around this would be to encapsulate `should_run_in_parallel` into a class, but since
    # it's a singleton, it ends up being functionally no different to a global anyway.
    global _PARALLEL_IGNORE_USER_SETTINGS  # noqa: PLW0603

    should_run_in_parallel.cache_clear()
    previous, _PARALLEL_IGNORE_USER_SETTINGS = _PARALLEL_IGNORE_USER_SETTINGS, True
    try:
        yield
    finally:
        _PARALLEL_IGNORE_USER_SETTINGS = previous
        should_run_in_parallel.cache_clear()


@contextlib.contextmanager
def _parallel_override(value: bool):
    """A context manager within which :func:`should_run_in_parallel` will return the given
    ``value``.

    This is not a *complete* override; Qiskit will never attempt to parallelize if only a single
    process is available, and will not allow process-based parallelism at a depth greater than 1."""
    # The way around this would be to encapsulate `should_run_in_parallel` into a class, but since
    # it's a singleton, it ends up being functionally no different to a global anyway.
    global _PARALLEL_OVERRIDE  # noqa: PLW0603

    should_run_in_parallel.cache_clear()
    previous, _PARALLEL_OVERRIDE = _PARALLEL_OVERRIDE, value
    try:
        yield
    finally:
        _PARALLEL_OVERRIDE = previous
        should_run_in_parallel.cache_clear()


should_run_in_parallel.ignore_user_settings = _parallel_ignore_user_settings
should_run_in_parallel.override = _parallel_override


def parallel_map(task, values, task_args=(), task_kwargs=None, num_processes=None):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    This will parallelise the results if the number of ``values`` is greater than one and
    :func:`should_run_in_parallel` returns ``True``.  If not, it will run in serial.

    By default multiprocessing will be disabled on all supported platforms when calling this
    function due to issues mixing it with threading. You have to explicitly opt-in to use
    multiprocessing with this function by either explicitly calling
    :func:`multiprocessing.set_start_method` to either ``"fork"`` or ``"forkserver"``,
    setting the environment variable ``QISKIT_PARALLEL=TRUE``, or setting
    ``parallel`` in your user configuration file. You can find more details
    on these configuration options here:
    https://quantum.cloud.ibm.com/docs/guides/configure-qiskit-local

    Args:
        task (func): Function that is to be called for each value in ``values``.
        values (array_like): List or array of values for which the ``task`` function is to be
            evaluated.
        task_args (list): Optional additional arguments to the ``task`` function.
        task_kwargs (dict): Optional additional keyword argument to the ``task`` function.
        num_processes (int): Number of processes to spawn.  If not given, the return value of
            :func:`default_num_processes` is used.

    Returns:
        result: The result list contains the value of ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    Examples:

        .. plot::
           :include-source:
           :nofigs:

            import time
            from qiskit.utils import parallel_map
            def func(_):
                    time.sleep(0.1)
                    return 0
            parallel_map(func, list(range(10)));
    """
    task_kwargs = {} if task_kwargs is None else task_kwargs
    if num_processes is None:
        num_processes = default_num_processes()
    if len(values) < 2 or not should_run_in_parallel(num_processes):
        return [task(value, *task_args, **task_kwargs) for value in values]
    work_items = ((task, value, task_args, task_kwargs) for value in values)

    # This isn't a user-set variable; we set this to talk to our own child processes.
    previous_in_parallel = os.getenv("QISKIT_IN_PARALLEL", _IN_PARALLEL_ALLOW_PARALLELISM)
    os.environ["QISKIT_IN_PARALLEL"] = _IN_PARALLEL_FORBID_PARALLELISM
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            return list(executor.map(_task_wrapper, work_items))
    finally:
        os.environ["QISKIT_IN_PARALLEL"] = previous_in_parallel
