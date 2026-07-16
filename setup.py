# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Qiskit setup file."

import os
import warnings
from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Most of this configuration is managed by `pyproject.toml`.  This only includes the extra bits to
# configure `setuptools-rust`, because we do a little dynamic trick with the debug setting, and we
# also want an explicit `setup.py` file to exist so we can manually call
#
#   python setup.py build_rust --inplace --release
#
# to make optimized Rust components even for editable releases, which would otherwise be quite
# unergonomic to do otherwise.


# Check for a default build profile from the environment (`--release` or `--debug` flags to
# `build_rust` override this default).  If not present, we also check if `RUST_DEBUG=1` for
# convenience, since we did that (undocumented) until Qiskit 2.4.
if (build_profile := os.getenv("QISKIT_BUILD_PROFILE", None)) is None:
    rust_debug = os.getenv("RUST_DEBUG", None) == "1" or None
else:
    match build_profile.lower():
        case "debug":
            rust_debug = True
        case "release":
            rust_debug = False
        case _:
            warnings.warn(
                f"QISKIT_BUILD_PROFILE set to unknown value '{build_profile}'."
                " Valid values are 'debug' and 'release'."
            )
            rust_debug = None

# If QISKIT_NO_CACHE_GATES is set then don't enable any features while building
#
# TODO: before final release we should reverse this by default once the default transpiler pass
# is all in rust (default to no caching and make caching an opt-in feature). This is opt-out
# right now to avoid the runtime overhead until we are leveraging the rust gates infrastructure.
if os.getenv("QISKIT_NO_CACHE_GATES") == "1":
    features = []
else:
    features = ["cache_pygates"]

if os.getenv("QISKIT_BUILD_WITH_MIMALLOC") == "1":
    features.append("mimalloc")


setup(
    rust_extensions=[
        RustExtension(
            "qiskit._accelerate",
            "crates/pyext/Cargo.toml",
            binding=Binding.PyO3,
            debug=rust_debug,
            features=features,
            generated_files={"include": "qiskit.capi", "_ctypes.py": "qiskit.capi"},
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
