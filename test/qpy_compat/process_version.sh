#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
set -e
set -x

function usage {
    echo "usage: ${BASH_SOURCE[0]} -p /path/to/qiskit/python <package> <version> <python_version>" 1>&2
    exit 1
}

python="python"
while getopts "p:" opt; do 
    case "$opt" in
        p)
            python="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done
shift "$((OPTIND-1))"
if [[ $# != 3 ]]; then
    usage
fi

# `package` is the name of the Python distribution to install (qiskit or qiskit-terra). `version` is
# the source version: the release with which to generate qpy files with to load with the version
# under test. 'python_version' is the (compatible) python version with which to run qiskit within the docker image,
# in the case where docker is used.

package="$1"
version="$2"
python_version="$3"

our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
cache_dir="$(pwd -P)/qpy_cache/$version"
venv_dir="$(pwd -P)/venvs/$package-$version"

# Use the updated constraints file for qiskit >= 2.5 (numpy >= 2.0.0 requirement).
constraints_file="qpy_test_constraints.txt"

major=${version%%.*}
rest=${version#*.}
minor=${rest%%[^0-9]*}

if (( major > 2 || (major == 2 && minor >= 5) )); then
    constraints_file="qpy_test_constraints25.txt"
fi

if [[ ! -d $cache_dir ]] ; then
    docker build -t $package:$version --build-arg PYTHON_VERSION=$python_version --build-arg PACKAGE_NAME=$package --build-arg PACKAGE_VERSION=$version --build-arg CONSTRAINTS_FILE=$constraints_file .
    mkdir -p "$cache_dir"
    pushd "$cache_dir"
    echo "Generating QPY files with $package==$version"
    # If the generation script fails, we still want to tidy up before exiting.
    docker run --rm -v "${our_dir}":/work/src -v "$PWD":/work -w /work $package:$version python src/test_qpy.py generate --version="$version" || { docker rmi $package:$version; exit 1; }
    docker rmi $package:$version
    docker builder prune -f
else
    echo "Using cached QPY files for $version"
    pushd "${cache_dir}"
fi
echo "Loading qpy files from $version with dev Qiskit"
"$python" "${our_dir}/test_qpy.py" load --version="$version"
popd
