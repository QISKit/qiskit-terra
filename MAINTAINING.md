# Maintainers Guide

This document defines a *maintainer* as a contributor with merge privileges.
The information detailed here is mostly related to Qiskit releases and other internal processes.


## Package Version

The version of the Qiskit package and crates is mentioned in a few places:

* `qiskit/VERSION.txt` for defining the Python package and docs
* `Cargo.toml` for defining the Rust crates
* `crates/bindgen/include/qiskit/version.h` for defining the C header file
* `docs/release_notes.rst` for configuring the release-notes documentation build
* `.mergify.yml` (implicitly via a branch name) for configuring where Mergify targets backports

In principle, the first three version numbers should be the same at all times.
However, the different languages have different conventions about formatting.

The `docs/release_notes.rst` version (in the `:earliest-version:` directive to `reno`) should match the git tag of the earliest release in the series (including pre-releases).
We use Python version-number formatting for our git tags.

### Version-number formatting

Python and Rust use different conventions for pre-release suffixes to the version.
The punctuation (or lack of) separating the main number and the suffix is important.

| Release level      | Python example | Rust or C example |
|--------------------|----------------|-------------------|
| stable             | 2.3.0          | 2.3.0             |
| release candidate  | 2.3.0rc1       | 2.3.0-rc1         |
| beta               | 2.3.0b1        | 2.3.0-beta1       |
| development        | 2.3.0.dev0     | 2.3.0-dev         |

In C the version string is custom and freeform, but we have a test that checks that it matches the Rust one.


### Updating the version number

The package version stored into the repository should be changed as follows:

- on `main`, the package version should almost always have a `dev` suffix and the version number should be the major/minor that is under development on `main`.
  For example, while 2.2.x is the current active release series of Qiskit, the version number on `main` should be `2.3.0.dev0`.

- on a stable branch, the version number should be whatever the most recent release on the stable branch was; it is incremented as part of the release process.
  For example, the `stable/2.3` branch is created from the commit that bumps the version number to `2.3.0rc1`.

  The `:earliest-version:` number in `docs/release_notes.rst` should be the earliest tag (including pre-releases) in that minor series.
  For example, `stable/2.2`'s earliest release is `2.2.0b1`, whereas `stable/2.1`'s is `2.1.0rc1`.

The procedure for a new minor-version release, with respect to version numbers is:

1. on `main`, push a PR that bumps the version from `2.2.0.dev0` to `2.2.0rc1` (and moves the loose release notes into `releasenotes/notes/2.2`, and then do the rest of the release process)
2. `qiskit-bot` will create a `stable/2.2` branch from that commit, since that's the one you should tag.
3. on `main`, immediately push a PR that bumps the version to `2.3.0.dev0` to open development on the 2.3 series, including updating `.mergify.yml` to backport to the new stable branch.

You will need to run `cargo build` as part of a version-bump commit to propagate the changes in `Cargo.toml` to `Cargo.lock`.


## Stable Branch Policy

The stable branch is intended to be a safe source of fixes for high-impact
bugs and security issues that have been fixed on `main` since a
release. When reviewing a stable branch PR, we must balance the risk
of any given patch with the value that it will provide to users of the
stable branch. Only a limited class of changes are appropriate for
inclusion on the stable branch. A large, risky patch for a major issue
might make sense, as might a trivial fix for a fairly obscure error-handling
case. A number of factors must be weighed when considering a
change:

-   The risk of regression: even the tiniest changes carry some risk of
    breaking something, and we really want to avoid regressions on the
    stable branch.
-   The user visibility benefit: are we fixing something that users might
    actually notice, and if so, how important is it?
-   How self-contained the fix is: if it fixes a significant issue but
    also refactors a lot of code, it's probably worth thinking about
    what a less risky fix might look like.
-   Whether the fix is already on `main`: a change must be a backport of
    a change already merged onto `main`, unless the change simply does
    not make sense on `main`.


### Backporting

When a PR tagged with `stable backport potential` is merged, or when a
merged PR is given that tag, the [Mergify bot](https://mergify.com) will
open a PR to the current stable branch.  You can review and merge this PR
like normal.


## Documentation Structure

Qiskit documentation is structured to keep as much of the content as possible within [docstrings](https://peps.python.org/pep-0257/#what-is-a-docstring).
This approach makes it easier to add or update documentation during development, since most of it lives close to the code being modified.
These docstrings are then pulled into [the API Reference section](https://quantum.cloud.ibm.com/docs/api/qiskit) of [quantum.cloud.ibm.com/docs](https://quantum.cloud.ibm.com/docs/).

Refer to [`qiskit_sphinx_theme` docs](https://qiskit.github.io/qiskit_sphinx_theme/apidocs/index.html) for how to create and
write effective API documentation, such as setting up the RST files and docstrings.


If changes you are making affect non-API reference content in [quantum.cloud.ibm.com/docs](https://quantum.cloud.ibm.com/docs),
you can open an issue (or better yet a PR) to update the relevant page in [Qiskit/documentation](https://github.com/Qiskit/documentation).
You can also use this repo to suggest or contribute brand new content beyond updates to the API reference.


## The release manager role

The processes around tracking that the correct code goes into a release in a timely manner, and releasing the package when it is time is the responsibility of the **release manager**.

Each release is tracked through a milestone that aggregates all associated issues and PRs.
This milestone should be continuously groomed throughout the release cycle. Additionally, at the release manager's discretion, a GitHub project board may be used to help prioritize tasks and monitor overall progress.
During the release cycle, the release manager is responsible for:

- Tracking the status of the remaining issues.

- Determining when issues/PRs need to be pushed to a later release, and which are high priority and the release should block on.
  Discuss with the team; you don't need to make all the decisions, you're just responsible for making sure
  the decisions are made.

- Making sure there are sufficient coders and reviewers assigned to each PR, and checking up on them to make sure they're progressing.

When we're getting close to release day, **do not** try and race through large and non-blocking PRs.
The release cycle for major and minor releases is regular so these can wait, and if it is a bugfix, we are free to make another patch release quickly.
Non-blocking issues with no associated PR within a day of the release
should be pushed for further release.

## How to release Qiskit

The precise steps depend on what kind of release you are making.
The choices are:

* First release candidate (e.g. `2.4.0rc1`)
* First public release (e.g. `2.4.0`)
* Follow-on patch release (e.g. `2.4.0rc2` or `2.4.1`)

We also occasionally do one-off "beta" releases (e.g. `1.3.0b1`) as demo versions for specific events.
These don't have a formal release process, because they tend to be highly ad-hoc.

Common assumptions in all commands in release recipes:

* The Qiskit/qiskit remote is called `upstream`.  If not: replace `upstream` with your name for the
  remote every place that it appears in every command.

* The `upstream` remote is up to date before running any commands in any recipe.  If not: run
  `git fetch --tags upstream`.

### Recipe for a first release candidate (`2.4.0rc1`)

These instructions will all use `2.4.0rc1` as the example version number being released, so adjust all version numbers accordingly for your release.
You will also see references to:

* `stable/2.3` (old stable branch)
* `stable/2.4` (new stable branch)
* `2.3.0rc1` (old-version release candidate)
* `2.4.0.dev0` (current development version)
* `2.5.0.dev0` (next development version)

that will all need to be adjusted in a suitable manner.

**Steps of the process**:

1. Check all P0 issues and PRs for the release are resolved.
2. Create the "release" PR on `main` that ([follow-along example for `2.4.0rc1`](https://github.com/Qiskit/qiskit/pull/15837)):

   * moves release notes from backported PRs into `releasenotes/notes/2.3` (the old stable folder)
     <details>
     Assumptions:

     * you have checked out the branch to make a PR;

     * the old stable branch is `stable/2.3`.

     ```bash
     notes=($(git diff --name-only ...upstream/stable/2.3 -- ':(glob)releasenotes/notes/*.yaml'))
     for note in "${notes[@]}"; do git mv -k "$note" releasenotes/notes/2.3/; done
     ```
     </details>

   * moves release notes from the new feature version into `releasenotes/notes/2.4` (the new stable folder)
     <details>
     Assumptions: you are on your PR branch.

     ```bash
     mkdir -p releasenotes/notes/2.4
     git mv releasenotes/notes/*.yaml releasenotes/notes/2.4/
     ```
     </details>

   * bumps the package-defining version numbers from the dev version (`2.4.0.dev0`) to the release version (`2.4.0rc1`)
     <details>
     This should bump only the places that actually specify the version of the package and not any
     repository-automation or documentation systems.

     See [Package Version](#package-version) at the top of this file for the up-to-date list.  It's
     just the Python package, Rust crates and C API numbers that need bumping in this PR.
     </details>

   * updates all Rust-space build dependencies in `Cargo.lock`
     <details>
     Assumptions: you are on your PR branch.

     In principle, the recipe is
     ```bash
     cargo update
     ```
     but this is unreliable.  See [Running `cargo update`](#running-cargo-update) for more detail.
     </details>

   * is labelled [ci: test wheels](https://github.com/Qiskit/qiskit/labels/ci:%20test%20wheels) in the GitHub web interface
     <details>
     This causes CI to run the wheel-build workflows in dry-run mode, which will show you any
     potential failures that might appear when you actually try to release.

     This is optional, it's just likely to save you time later if there are problems.
     </details>

   * updates the QPY version-support table in the `qiskit.qpy` docstring, including patch releases
     <details>
     The supported versions are everything between `qiskit.qpy.QPY_COMPATIBLITY_VERSION` and
     `QPY_VERSION`, inclusive.

     See https://github.com/Qiskit/qiskit/pull/16034, for example.
     </details>

   * updates the C API "slots check" file
     <details>
     This updates the C-API slots lint job to use this new release as the base
     for semver compatibility checks for new PR.

     ```bash
     cargo run -p qiskit-bindgen-cli -- show-slots > capi_slots.txt
     ```
     </details>

3. Tag the resulting PR after merge as `2.4.0rc1`, and push it. (Detail: [How to tag and release a complete version](#how-to-tag-and-release-a-complete-version).)

4. Create the new stable branch (`stable/2.4`) from the same commit, and push it.
   <details>
   Assumptions:

   * you have `main` checked out locally, and it is updated so the `HEAD` is the PR from step 2 and
     the tag from step 3.

   * you are in the `terra-core` group on GitHub, so you have permissions to push branches (if not:
     ask Jake or Matt about it).

    * you have just released `2.4.0rc1` (if not: adjust the `stable/2.4` branch name appropriately).

   ```bash
   git branch stable/2.4 2.4.0rc1
   git push --set-upstream upstream stable/2.4
   ```
   </details>

5. Create the "open new development" PR on `main` that ([follow-along example for `2.5.0.dev0`](https://github.com/Qiskit/qiskit/pull/15840)):

   * bumps the package-defining version numbers from the rc version (`2.4.0rc1`) to the new dev version (`2.5.0.dev0`)
     <details>
     This is the same as the equivalent package-bump version in step 2; it's the same version
     numbers that need updating.

     After updating the version numbers, pull them into `Cargo.lock` such as with `cargo check`.
     </details>

   * bumps version-number references to the old stable (`stable/2.3` or `2.3.0rc1`) to the new one (`stable/2.4` or `2.4.0rc1`)
     <details>
     This is all the version numbers you didn't update in previous steps.

     See [Package Version](#package-version) at the top of this file for the up-to-date list.  It's
     things like the Mergify backport configuration and the documentation "earliest version" numbers
     that need updating.
     </details>

### Recipe for a first public version (`2.3.0`)

These instructions use `2.3.0` as the example number, and the corresponding stable branch
`stable/2.3`.  Adjust the numbers as appropriate.

**Steps of the process**:

1. Check all P0 issues and PRs for the release are resolved.
2. Create the "release" PR on `stable/2.3` that ([follow-along example for
   `2.3.0`](https://github.com/Qiskit/qiskit/pull/15514)):

   * adds a `prepare-2.3.0` release note that contains a `prelude` section
     <details>
     Create the release note with:

     ```bash
     reno new --edit prepare-2.3.0
     ```

     Delete the entire template; we only need a section which isn't in the template.  Replace it
     with

     ```rst
     ---
     prelude: |
        Qiskit v2.3.0 is a new feature release of the Qiskit SDK.

        The rest of the release note will go here.
     ```

     Use the prelude to advertize the primary new features of the release.  Aim for one paragraph
     each for approximately three headline features.  Check with the team if you are on unsure what
     should be in here.
     </details>

   * checks all the release notes for this version for grammar and correctness
     <details>
     You only need to look at release notes that are loose in `releasenotes/notes` or in
     `releasenotes/notes/2.3`.

     Ask the docs team to help check grammar and spelling, and do whatever they say after you've
     checked the technical details are correct; they are the arbiters of our public
     written-documentation style, not us.

     You may want to build the documentation locally to help spot errors.

     Things to check for:

     * All Sphinx cross-references will link correctly, and will have useful link text. For example,
       referring to `` :func:`~qasm2.load` `` is unlikely to be legible for readers since the
       display text will be `load` and they won't know which module you mean.  Try
       `` :func:`.qasm2.load` `` instead.

     * All "sections" are valid entries in `releasenotes/config.yaml`, and use as tight a scope as
       possible.  For example, nothing should use the base `features` section, but instead use
       `features_c` or `features_qasm`, or similar.

     * Each bullet point of each release note can be read completely in isolation with no additional
       context.

     * Each bullet point is as concise as is reasonable.  We want to give people a summary version,
       not the full detail; there are a lot of release notes on the page.

     * Feature release notes _may_ have code examples, but keep them short.  Prefer to link to API
       documentation with worked examples instead.

     * Bugfix release notes should be about two sentences and should _not_ have code examples.
       Prefer to link to suitable GitHub issues explaining the bug that was fixed.
     </details>

   * bumps the package-version defining numbers from the rc (`2.3.0rc1`) to the final (`2.3.0`)

     <details>
     This should bump only the places that actually specify the version of the package and not any
     repository-automation or documentation systems.

     See [Package Version](#package-version) at the top of this file for the up-to-date list.  It's
     just the Python package, Rust crates and C API numbers that need bumping in this PR.

     Run `cargo check` locally to propagate Rust version-number updates to `Cargo.lock`.
     </details>

3. Tag the resulting PR after merge as `2.3.0`, and push it. (Detail: [How to tag and release a complete version](#how-to-tag-and-release-a-complete-version).)


### Recipe for a follow-on patch version (`2.4.0rc2` or `2.4.1`)

This is basically a simpler version of the "first public release" recipe.  We are using `2.3.1` as
the example version number; adjust as appropriate.

**Steps of the process**:

1. Check that all PRs that are intended to go into this release have been fully backported.
2. Create the "release" PR on `stable/2.3` that ([follow-along example for
   `2.3.1`](https://github.com/Qiskit/qiskit/pull/15803)):

   * adds a `prepare-2.3.1` release note that contains a `prelude` section
     <details>
     Create the release note with:

     ```bash
     reno new --edit prepare-2.3.1
     ```

     Replace the entire file with:

     ```rst
     ---
     prelude: |
        Qiskit v2.3.1 is a new bugfix release of the Qiskit SDK.
     ```
     </details>

   * checks all the release notes for this version for grammar and correctness
     <details>
     You only need to check new backported release notes.  If your tags are locally up-to-date,
     you can get a list of the release notes that need checking with:

     ```bash
     git diff --name-only 2.3.0...stable/2.3 -- releasenotes/notes
     ```

     Adjust the base tag (`2.3.0`) and stable branch (`stable/2.3`) as appropriate.
     </details>

   * bumps the package-version defining numbers from the previous (`2.3.0`) to the final (`2.3.0`)
     <details>
     This should bump only the places that actually specify the version of the package and not any
     repository-automation or documentation systems.

     See [Package Version](#package-version) at the top of this file for the up-to-date list.  It's
     just the Python package, Rust crates and C API numbers that need bumping in this PR.

     Run `cargo check` locally to propagate Rust version-number updates to `Cargo.lock`.
     </details>
 
   You can skip all the "release notes" steps if you are releasing a follow-on release candidate and
   are pressed for time.

3. Tag the resulting PR after merge as `2.3.1`, and push it. (Detail: [How to tag and release a complete version](#how-to-tag-and-release-a-complete-version).)

### How to tag and release a complete version

This is the recipe for making a tag and pushing it, for _any_ version of Qiskit.  It's the same for
first release candidates, first public versions and all subsequent patches.

Assumptions:

* the commit that will become the release is merged to the correct branch (`main` for first release
  candidates, `stable/*` for all other versions).
* you have got the correct commit checked out locally.
* you have a GPG key registered with `git` (if not: consider configuring one in the future, and
  in the mean time remove the `--sign` option from `git tag` in the recipe)
* you are in the `terra-release` group on GitHub, so you have permissions to push tags (if not:
  ask Jake or Matt about it).
* the version number is `2.4.0rc1` (if not: adjust all instances of the version number
  appropriately, including in the tag message).

Recipe:

1. Make the tag

   ```bash
   git tag --sign -m "Qiskit 2.4.0rc1" 2.4.0rc1
   ```

2. Verify the tag is correct (check the tagged commit is the PR from step 2, and check
   `upstream/main` points to it too):

   ```bash
   git show 2.4.0rc1
   ```

3. Push the tag to `upstream`.  **This performs the release** (though it must still be approved
   by a second maintainer before it will be deployed).
   ```bash
   git push upstream 2.4.0rc1
   ```

4. Follow the progress of the build and deployment in the relevant workflow run linked in
   https://github.com/Qiskit/qiskit/actions/workflows/wheels.yml.  When a "deploy" step is
   reached, all other (not you!) maintainers will receive a notification asking them to "approve"
   the deployment.  They must:

   * verify the tag has the correct version number
   * verify the tag points to the correct commit
   * assuming all is fine, approve the request to deploy the artifacts,
     confirming the two points above were checked. **This finalizes the
     release and deploys it.**

5. Announce the release.

   Places to anounce:

   * IBM-internal Slack channels (ask if you are unsure).
   * [Public Qiskit Slack channels](https://qiskit.enterprise.slack.com): `#announcements` (final
     releases), `#roadmap-announcements` (all), `#qiskit-dev` (all).

   Look at previous messages in the relevant channels for examples.


## Running `cargo update`

We occasionally want to update all transitive Rust dependencies in the project.  Typically this is
done at the first release candidate of a new feature release, but you can do it at any time.

_In theory_, you should just be able to run (using a `cargo` from the Rust version matching
`rust-version` in `Cargo.toml`):
```bash
cargo update --verbose
```
commit the result and go about your life.  In practice, Qiskit's Rust-space dependency story is
messy and you may encounter problems, so watch the output of that command and:

* Check for any dependencies that produced a warning about their MSRV going above Qiskit's limit.
  This should only happen in transitive dependencies; `nalgebra` is a common culprit via `numpy`,
  but this can change.

  You can "downgrade" a particular package with
  ```bash
  cargo update nalgebra@0.34.1 --precise 0.33.2
  ```
  which downgrades all instances of `nalgebra==0.34.1` to version `0.33.2` instead.

* Check the package still builds.  If you get reams of errors such as:
  ```text
  hashbrown::HashMap<&str, usize> cannot be converted to a Python object
  ```
  or other things about "trait methods not satisfied" or similar, the problem might be
  dependency-version coherence.

  There are two related problems here:

  * PyO3 has features that implement its Python-conversion methods for dependencies like
    `hashbrown`.  This only works if Qiskit and PyO3 use the _same_ version of `hashbrown`.  `cargo`
    is not aware of this constraint, and will frequently attempt to bump the version of `hashbrown`
    that PyO3 is compiled against, while keeping Qiskit's locked.  You will need to manually
    downgrade the version that PyO3 uses, to match Qiskit's.  You might have to edit `Cargo.lock`
    manually to achieve this (just look for the line like `hashbrown 0.16.1` in the
    `dependencies` array of the `[[package]]` with `name = "pyo3"` and modify the number) because
    the `cargo update --precise` trick from above is not precise enough.

    Similarly, `numpy` depends on PyO3, and the version of PyO3 it uses needs to match the version
    that Qiskit uses.

  * Some libraries use external types from dependencies in their public interfaces, so they need to
    Qiskit and that library need to use the same version of the dependency.  This is most common
    with `rustworkx` using `hashbrown` types.  Similar to the previous bullet point, you might have
    to manually edit `Cargo.lock` to fix the situation.

Hopefully, later versions of `cargo` will give us better tools to deal with these, but for now the
process can be quite manual.
