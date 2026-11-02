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

ifneq ($(OS), Windows_NT)
	OS := $(shell uname -s)
endif

ifeq ($(QISKIT_BUILD_WITH_MIMALLOC), 1)
	MIMALLOC := --features=mimalloc
endif

.PHONY: default env lint style black test test_randomized pytest pytest_randomized test_ci coverage coverage_erase clean

default: style lint test ;

# Dependencies need to be installed on the Anaconda virtual environment.
env:
	if test $(findstring qiskitenv, $(shell conda info --envs | tr '[:upper:]' '[:lower:]')); then \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	else \
		conda create -y -n Qiskitenv python=3; \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	fi;

lint:
	ruff check qiskit test tools setup.py
	tools/verify_headers.py qiskit test tools
	tools/find_optional_imports.py
	tools/find_stray_release_notes.py
	tools/verify_images.py
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

style:
	black --check qiskit test tools setup.py docs/conf.py

black:
	black qiskit test tools setup.py docs/conf.py

# Use the -s (starting directory) flag for "unittest discover" is necessary,
# otherwise the QuantumCircuit header will be modified during the discovery.
test:
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================
	python3 -m unittest discover -s test/python -t . -v
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================

# Use pytest to run tests
pytest:
	pytest test/python

# Use pytest to run randomized tests
pytest_randomized:
	pytest test/randomized

test_ci:
	QISKIT_TEST_CAPTURE_STREAMS=1 stestr run

test_randomized:
	python3 -m unittest discover -s test/randomized -t . -v

coverage:
	coverage run --source qiskit -m unittest discover -s test/python -q
	coverage report

coverage_erase:
	coverage erase

clean: coverage_erase ;

# ==============================================================================
# Variables that can be set/modified to modify the C builds.
# ==============================================================================

# Output directories.
C_DIR_OUT=dist/c
C_DIR_OUT_LIB=$(C_DIR_OUT)/lib
C_DIR_OUT_INCLUDE=$(C_DIR_OUT)/include
C_DIR_TEST=test/c
C_DIR_TEST_BUILD=test/c/build

# Input directories
C_DIR_CARGO_TARGET=target
C_DIR_CRATES=crates

# ==============================================================================
# Variables that we derive from the settings above.
# ==============================================================================

ifeq ($(OS), Windows_NT)
	C_LIB_CARGO_FILENAME=qiskit_cext.dll
else ifeq ($(shell uname), Darwin)
	C_LIB_CARGO_FILENAME=libqiskit_cext.dylib
else
	# ... probably.
	C_LIB_CARGO_FILENAME=libqiskit_cext.so
endif

C_LIBQISKIT_OUT=$(C_DIR_OUT_LIB)/$(subst _cext,,$(C_LIB_CARGO_FILENAME))

# ==============================================================================
# Recipes for the C components.
# ==============================================================================

.PHONY: cformat fix_cformat ctidy
# Run clang-format (does not apply any changes)
cformat:
	bash tools/run_clang_format.sh
# Apply clang-format changes
fix_cformat:
	bash tools/run_clang_format.sh apply

# Run clang tidy, which currently only checks for `printf` occurences
# This currently only checks for files in test/c/*.c. We could turn this into
# a bash script, like for clang-format, which globs for .c files, but we only really care to
# run this print check on the test files at the moment.
ctidy: cheader
	clang-tidy $(C_DIR_TEST)/*.c -- -std=c11 -I$(C_DIR_OUT_INCLUDE)

# Abstraction over calling Cargo to build the C extension in "standalone" C
# mode.  This _also_ builds the C header file as a side-effect into
# `target/qiskit.h`.  Recipes that use this as a prerequisite should ensure they
# set `C_LIB_CARGO_FLAGS` to choose the build profile.  The `C_LIB_RUSTC_FLAGS`
# variable can also be set to add additional logic (like coverage instructions).
#
# Typically, downstream recipes should depend on `build-clib-release` or `build-clib-dev`
# instead.
.PHONY: build-clib build-clib-release build-clib-dev
build-clib:
	cargo rustc -p qiskit-cext ${MIMALLOC} --crate-type cdylib ${C_LIB_CARGO_FLAGS} -- ${C_LIB_RUSTC_FLAGS}
build-clib-release: C_LIB_CARGO_FLAGS=--release
build-clib-release: build-clib
build-clib-dev: C_LIB_CARGO_FLAGS=--profile dev
build-clib-dev: build-clib

# Catch-all directory-creation rule.
$(C_DIR_OUT_LIB):
	mkdir -p $@

.PHONY: cheader
cheader:
	cargo run -p qiskit-bindgen-cli -- install -c $(C_DIR_CRATES)/cext -o $(C_DIR_OUT_INCLUDE)
# `clib` and `clib-dev` are conflicting rules - they both attempt to "install" the
# shared library into the output `lib` directory, but they differ between release
# and dev mode.
.PHONY: clib
clib: build-clib-release | $(C_DIR_OUT_LIB)
	cp $(C_DIR_CARGO_TARGET)/release/$(C_LIB_CARGO_FILENAME) $(C_LIBQISKIT_OUT)
.PHONY:
clib-dev: build-clib-dev | $(C_DIR_OUT_LIB)
	cp $(C_DIR_CARGO_TARGET)/debug/$(C_LIB_CARGO_FILENAME) $(C_LIBQISKIT_OUT)
.PHONY: c
c: cheader clib

.PHONY: ctest
# Use ctest to run C API tests.
ctest: cheader build-clib-dev
# `-S` specifies the source (including the `CMakeLists.txt` file, `-B` is where
# to put the build files, including the generated CMake stuff.  See the
# `CMakeLists.txt` file for the build variables.
#
# The `CMAKE_FLAGS` argument is to allow user insertion of additional flags, such
# as to override the `CMAKE_C_STANDARD` line.
	cmake -S$(C_DIR_TEST) -B$(C_DIR_TEST_BUILD) \
		$(CMAKE_FLAGS) \
		-DCARGO_LIB_DIR=$(abspath $(C_DIR_CARGO_TARGET))/debug \
		-DQISKIT_INCLUDE_PATH=$(abspath $(C_DIR_OUT_INCLUDE))
# Actually build the test.
	cmake --build $(C_DIR_TEST_BUILD)
# -V ensures we always produce a logging output to indicate the subtests
# -C Debug is needed for windows to work, if you don't specify Debug (or
# Release) explicitly ctest doesn't run on windows
	ctest -V -C Debug --test-dir $(C_DIR_TEST_BUILD)

.PHONY: ccoverage
ccoverage: C_LIB_RUSTC_FLAGS=-Cinstrument-coverage
ccoverage: ctest

.PHONY: qiskit-pyo3-ffi
qiskit-pyo3-ffi:
	rm -rf dist/rust/qiskit-pyo3-ffi
	cargo run -p qiskit-bindgen-cli -- generate-pyo3 --cext-path crates/cext --output-path dist/rust/qiskit-pyo3-ffi

.PHONY: cclean
cclean:
	rm -rf $(C_DIR_OUT) $(C_DIR_TEST_BUILD) $(C_INCLUDE_FILES_ABS_GENERATED)
	cargo clean --package qiskit-cext
