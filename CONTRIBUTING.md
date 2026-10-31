# Contributing

## Quickstart

Qiskit is an open-source project committed to bringing quantum computing to people of all
backgrounds.  We accept contributions using standard GitHub issue and pull-request processes.
When interacting with the repository, please ensure you abide by the following rules:

1. All interactions must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
2. If you intend to use generative AI, ensure you abide by the [Generative AI
   policy](docs/devguide/contributing.md#use-of-generative-ai).
3. You must be authorized to sign our [Contributor License Agreement (CLA)](https://qisk.it/cla) for
   all code contributions you make.

### Building Qiskit from source

> [!NOTE]
> More detail: [Installing Qiskit from source](docs/devguide/contributing.md#installing-qiskit-from-source).

Qiskit is a mixed Rust/Python project.  You must have Python available, and a Rust compiler
installed.  For Rust, we suggest using [Rustup](https://rustup.rs/) to manage your compiler
toolchains.

The easiest way to build the Qiskit Python packages is to run the standard command:

```bash
pip install -e .
```

This will build Qiskit in a debug mode, which is much slower than release-mode builds.

Unlike pure-Python projects, you will need to manually rebuild the Rust components if you make
changes to the Rust files, including if you update your local `git` checkout of Qiskit.  You can run
the same `pip install -e .` command to do this.

To build the C library in stand-alone release mode for distribution, use the GNU Makefile rule `make
c`.  This will place the C artifacts in `dist/c`.  Use `make cheader clib-dev` if you need a debug
build.

### Running local tests

> [!NOTE]
> More detail: [Testing](docs/devguide/contributing.md#testing).

Qiskit's test suite has several components.  We have unit and integration tests in Python, Rust and
C, and we use various static-analysis tools.

- **Python**: The simplest way to run the tests is to use `tox`, which is a runner you install
  separately, such as with `pipx` or `uv tool run`/`uvx`.  Use your preferred tool to install `tox`,
  then you can run the test suite with `tox -e py`.

- **Rust**: Use the standard `cargo test` machinery.

- **C**: You need CMake and GNU Make.  Run `make ctest`.

- **Formatting**: We use `black` for Python code (see the `lint` dependency-group in
  `pyproject.toml`), `rustfmt` for Rust, and `clang-format` for C.  You can use `tox -e black` to
  autoformat Python code, `cargo fmt` for Rust, and `make cformat` for C.

- **Linting**: Python uses `ruff`, Rust uses `clippy`, C uses `clang-tidy` and various compiler
  warnings, and there are a few custom lints.  You can run everything except the C lints with `tox
  -e lint`, and the C lints with `make ctidy`.

### Making pull requests

> [!NOTE]
> More detail: [Issues and pull requests](docs/devguide/contributing.md#issues-and-pull-requests).

When making a pull request, you should follow the [Pull request author
checklist](docs/devguide/contributing.md#pull-request-author-checklist).  You must fill in the PR
template that will be presented.

We prefer that any proposed changes are discussed on an issue before making a pull request.  Please
try to take care in all communications to be concise; please avoid needless detail and technical deep-dives, _especially_ if using LLM assistance.

We receive a lot of pull requests, so review might take a short while.  Please be patient.  You do
not need to press the "update branch" button unless there are merge conflicts.


## Detailed developer guide

We have more detailed documentation on various aspects of contributing and developing Qiskit.

- [General contribution guidelines](docs/devguide/contributing.md)
- [Deprecation policy](docs/devguide/deprecation.md)
- [Maintainers' guide](docs/devguide/maintaining.md)
