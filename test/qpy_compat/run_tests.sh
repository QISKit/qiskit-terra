#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
no_docker=false
if [[ " $@ " =~ " --no-docker " ]]; then
    no_docker=true
fi

set -e
set -o pipefail
set -x
shopt -s nullglob

# Set fixed hash seed to ensure set orders are identical between saving and
# loading.
export PYTHONHASHSEED=$(python -S -c "import random; print(random.randint(1, 4294967295))")
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
repo_root="$(realpath -- "$our_dir/../..")"

# First, prepare a wheel file for the dev version.  We install several venvs with this, and while
# cargo will cache some rust artefacts, it still has to re-link each time, so the wheel build takes
# a little while.

wheel_dir="$(pwd -P)/wheels"
python -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$repo_root"
all_wheels=("$wheel_dir"/*.whl)
qiskit_dev_wheel="${all_wheels[0]}"

# Now set up a "base" development-version environment, which we'll use for most of the backwards
# compatibility checks.
qiskit_venv="$(pwd -P)/venvs/dev"
qiskit_python="$qiskit_venv/bin/python"
python -m venv "$qiskit_venv"

# `packaging` is needed for the `get_versions.py` script.
"$qiskit_venv/bin/pip" install -c "$repo_root/constraints.txt" "$qiskit_dev_wheel" packaging "symengine<0.14" "sympy>1.3"

# Run all of the tests of cross-Qiskit-version compatibility.
if $no_docker; then
    "$qiskit_python" "$our_dir/get_versions.py" --no-docker | parallel -j 2 --halt soon,fail=1 --colsep=" " bash "$our_dir/process_version_with_venv.sh" -p "$qiskit_python"
else
    "$qiskit_python" "$our_dir/get_versions.py" | parallel -j 2 --halt soon,fail=1 --colsep=" " bash "$our_dir/process_version.sh" -p "$qiskit_python"
fi

# Test dev compatibility with itself.
dev_version="$("$qiskit_python" -c 'import qiskit; print(qiskit.__version__)')"
mkdir -p "dev-files/base"
pushd "dev-files/base"
"$qiskit_python" "$our_dir/test_qpy.py" generate --version="$dev_version"
"$qiskit_python" "$our_dir/test_qpy.py" load --version="$dev_version"
popd


# Test dev compatibility with all supported combinations of symengine between generator and loader.
# This will likely duplicate the base dev-compatibility test, but the tests are fairly fast, and
# it's better safe than sorry with the serialisation tests.

# Note that the constraint in the range of symengine versions is logically duplicated
# in `qpy_test_constraints.txt`
symengine_versions=(
    '>=0.11,<0.12'
    '>=0.13,<0.14'
)
symengine_venv_prefix="$(pwd -P)/venvs/dev-symengine-"
symengine_files_prefix="$(pwd -P)/dev-files/symengine-"

# Create the venvs and QPY files for each symengine version.
for i in "${!symengine_versions[@]}"; do
    specifier="${symengine_versions[$i]}"
    symengine_venv="$symengine_venv_prefix$i"
    files_dir="$symengine_files_prefix$i"
    python -m venv "$symengine_venv"
    "$symengine_venv/bin/pip" install -c "$repo_root/constraints.txt" packaging "$qiskit_dev_wheel" "symengine$specifier"
    mkdir -p "$files_dir"
    pushd "$files_dir"
    "$symengine_venv/bin/python" -c 'import symengine; print(symengine.__version__)' > "SYMENGINE_VERSION"
    "$symengine_venv/bin/python" "$our_dir/test_qpy.py" generate --version="$dev_version"
    popd
done

# For each symengine version, try loading the QPY files from every other symengine version.
for loader_num in "${!symengine_versions[@]}"; do
    loader_venv="$symengine_venv_prefix$loader_num"
    loader_version="$(< "$symengine_files_prefix$loader_num/SYMENGINE_VERSION")"
    for generator_num in "${!symengine_versions[@]}"; do
        generator_files="$symengine_files_prefix$generator_num"
        generator_version="$(< "$generator_files/SYMENGINE_VERSION")"
        echo "Using symengine==$loader_version to load files generated with symengine==$generator_version"
        pushd "$generator_files"
        "$loader_venv/bin/python" "$our_dir/test_qpy.py" load --version="$dev_version"
        popd
    done
done

# Test forward compatibility: Generate QPY files for all supported QPY format versions
# using the current Qiskit API, then verify that the current code can load them all.
# This tests that we can read all supported QPY format versions.
echo "Testing forward compatibility: generating QPY files for all supported format versions"

# Get supported QPY versions from qiskit/qpy/common.py
# QPY_VERSION is the latest, QPY_COMPATIBILITY_VERSION is the oldest supported
qpy_latest=$("$qiskit_python" -c "from qiskit.qpy.common import QPY_VERSION; print(QPY_VERSION)")
qpy_oldest=$("$qiskit_python" -c "from qiskit.qpy.common import QPY_COMPATIBILITY_VERSION; print(QPY_COMPATIBILITY_VERSION)")

echo "Supported QPY versions: $qpy_oldest to $qpy_latest"

# Generate QPY files for each supported format version
for qpy_ver in $(seq "$qpy_oldest" "$qpy_latest"); do   
    files_dir="$(pwd -P)/forward-qpy-files/qpy${qpy_ver}"
    mkdir -p "$files_dir"
    pushd "$files_dir"
    echo "Generating QPY v${qpy_ver} for version ${dev_version}"
    "$qiskit_python" "$our_dir/test_qpy.py" generate --qpy-version="$qpy_ver" --version="$dev_version"
    popd
done

echo "Getting forward compatibility versions..."
"$qiskit_python" "$our_dir/get_forward_compat_versions.py" | \
    parallel -j 2 --halt soon,fail=1 --colsep=" " bash "$our_dir/process_forward_version.sh"