#!/bin/bash
#
# Prepare an Ubuntu CI machine for running 'tox -e docs'.  Assumes that Python is available.

set -e
if [ -z $GITHUB_TOKEN ]; then
    echo "No github token provided"
    exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade tox

sudo "$(dirname "${BASH_SOURCE[0]}")/install_ubuntu_graphviz.sh"

# This command fetches the latest release of doxygen and its linux binaries
wget --header "Authorization: token $GITHUB_TOKEN" \
    https://github.com/doxygen/doxygen/releases/download/Release_1_15_0/doxygen-1.15.0.linux.bin.tar.gz

# The following commands install the binaries for doxygen
tar -zxvf ./doxygen-1.15.0.linux.bin.tar.gz
cd ./doxygen-1.15.0

# Run the remainder of the setup process
sudo make install
