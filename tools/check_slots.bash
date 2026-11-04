#!/bin/bash

set -e

repo_root="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$repo_root"

cargo run --quiet -p qiskit-bindgen-cli -- lint-slots --cext-path "$repo_root/crates/cext"
