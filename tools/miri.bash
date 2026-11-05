#!/bin/bash

set -eo pipefail

usage() {
    cat <<EOF
Usage
    miri.bash <miri-test-args>
        Run Miri with Qiskit's default configuration.
        Additional arguments are appended to the \`cargo miri test\` command.

    miri.bash -p
        Stop after running \`cargo miri setup\`.

    miri.bash -t
        Print the configured toolchain for Miri and exit.

    miri.bash -h
        Print this help message and exit.

Configuration
    The toolchain to use for Miri can be overridden by setting the
    \`RUSTUP_TOOLCHAIN\` environment variable.

    A default set of \`MIRIFLAGS\` are included.  More can be added by setting
    that environmental variable.
EOF
}

export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN-nightly-2025-08-11}"
# - `tree-borrows` is required for crossbeam components.
# - `symbolic-alignment-check` is extra checking.
# - `strict-provenance` is extra checking.
export MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-symbolic-alignment-check -Zmiri-strict-provenance ${MIRIFLAGS-}"

while getopts ':pth' opt; do
    case $opt in
        p) cargo miri setup; exit;;
        t) echo "$RUSTUP_TOOLCHAIN"; exit;;
        h) usage >&2; exit;;
        *) usage >&2; exit 1;;
    esac
done
shift "$((OPTIND - 1))"

root_cargo_toml="$(cargo locate-project --workspace --message-format=plain)"
root_cargo_lock="$(dirname "$root_cargo_toml")/Cargo.lock"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir" || :' EXIT
cp "$root_cargo_lock" "$tmpdir/Cargo.lock"

# Some of our dependencies aren't Miri-safe with their current release versions.
# These need overriding with known-good versions to run against until the
# Miri-safe versions are released and updated in our `Cargo.lock`.
cat >>"$tmpdir/Cargo.toml" <<EOF
[patch.crates-io]
crossbeam-epoch = { git = 'https://github.com/crossbeam-rs/crossbeam', rev = '9e859610' }
EOF

cargo_version="$(cargo --version)"
if [[ $cargo_version =~ ^cargo\ 1\.([0-9]+)\..*$ ]]; then
    cargo_minor="${BASH_REMATCH[1]}"
else
    echo "failed to parse cargo version: '${cargo_version}'" >&2
    exit 2
fi
# Prior to Cargo 1.97, the `resolver.lockfile-path` entry was unstable.  The
# option turned up in some Cargo (about 1.82?) that's older than our MSRV.
#
# We use a temporary `Cargo.lock` cloned from our checked-in one to avoid the
# patches from making the repository dirty.
cargo_lockfile_args=()
if [[ "$cargo_minor" -lt 97 ]]; then
    cargo_lockfile_args+=("-Zunstable-options" "--lockfile-path=$tmpdir/Cargo.lock")
else
    cat >>"$tmpdir/Cargo.toml" <<EOF
[resolver]
lockfile-path = "$tmpdir/Cargo.lock"
EOF
fi

cargo miri test "${cargo_lockfile_args[@]}" --config="$tmpdir/Cargo.toml" "$@"
