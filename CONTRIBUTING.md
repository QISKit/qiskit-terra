# Contributing

Qiskit is an open-source project committed to bringing quantum computing to
people of all backgrounds. This page describes how you can join the Qiskit
community in this goal.


## Contents
* [Before you start](#before-you-start)
* [Choose an issue to work on](#Choose-an-issue-to-work-on)
* [Set up Python virtual development environment](#set-up-python-virtual-development-environment)
* [Installing Qiskit from source](#installing-qiskit-from-source)
* [Issues and pull requests](#issues-and-pull-requests)
  * [Pull request author checklist](#pull-request-author-checklist)
  * [Code review](#code-review)
  * [Pull request merging checking](#pull-request-merging-checklist)
  * [Use of AI tools](#use-of-ai-tools)
* [Contributor Licensing Agreement](#contributor-licensing-agreement)
* [Changelog generation](#changelog-generation)
* [Release notes](#release-notes)
* [Testing](#testing)
  * [Qiskit's Python test suite](#qiskits-python-test-suite)
  * [Snapshot testing for visualizations](#snapshot-testing-for-visualizations)
  * [Testing Rust components](#testing-rust-components)
    * [Using a custom venv instead of tox](#using-a-custom-venv-instead-of-tox)
    * [Calling Python from Rust tests](#calling-python-from-rust-tests)
* [Style and Lint](#style-and-lint)
* [Building API docs locally](#building-api-docs-locally)
  * [Troubleshooting docs builds](#troubleshooting-docs-builds)
* [Development Cycle](#development-cycle)
  * [Branches](#branches)
  * [Release Cycle](#release-cycle)
* [Adding deprecation warnings](#adding-deprecation-warnings)
* [Using dependencies](#using-dependencies)
  * [Version support policy](#version-support-policy)
  * [Adding a requirement](#adding-a-requirement)
  * [Adding an optional dependency](#adding-an-optional-dependency)
  * [Checking for optionals](#checking-for-optionals)
* [Dealing with git blame ignore list](#dealing-with-the-git-blame-ignore-list)


## Before you start

If you are new to Qiskit contributing we recommend you do the following before diving into the code:

* Read the [Code of Conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md)
* Familiarize yourself with the Qiskit community (via [Slack](https://qisk.it/join-slack),
   [Stack Exchange](https://quantumcomputing.stackexchange.com), [GitHub](https://github.com/qiskit-community/feedback/discussions) etc.)


## Choose an issue to work on
Qiskit uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

* [good first issue](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, ideal for newcomers. They should already be fully scoped, with a clear approach outlined in the descriptions.
* [help wanted](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these issues are generally more complex than good first issues. They typically cover work that core maintainers don't currently have capacity to implement and may require more investigation/discussion. These are a great option for experienced contributors looking for something a bit more challenging.
* [short project](https://github.com/Qiskit/qiskit/issues?q=is%3Aopen+is%3Aissue+label%3A%22short+project%22) - these issues are bigger pieces of work that require greater time commitment. Good options for hackathons, internship projects etc.


## Set up Python virtual development environment

Virtual environments are used for Qiskit development to isolate the development environment
from system-wide packages. This way, we avoid inadvertently becoming dependent on a
particular system configuration. For developers, this also makes it easy to maintain multiple
environments (e.g. one per supported Python version, for older versions of Qiskit, etc.).

### Set up a Python venv

All Python versions supported by Qiskit include built-in virtual environment module
[venv](https://docs.python.org/3/tutorial/venv.html).

Start by creating a new virtual environment with `venv`. The resulting
environment will use the same version of Python that created it and will not inherit installed
system-wide packages by default. The specified folder will be created and is used to hold the environment's
installation. It can be placed anywhere. For more detail, see the official Python documentation,
[Creation of virtual environments](https://docs.python.org/3/library/venv.html).

```
python3 -m venv ~/.venvs/qiskit-dev
```

Activate the environment by invoking the appropriate activation script for your system, which can
be found within the environment folder. For example, for bash/zsh:


```
source ~/.venvs/qiskit-dev/bin/activate
```

Upgrade pip within the environment to ensure Qiskit dependencies installed in the subsequent sections
can be located for your system.  You need `pip>=25.1` to use the `--group` feature used to manage
developer dependency groups.

```
pip install -U pip
```

You then install Qiskit in editable mode using:

```
pip install -e .
```

Changes to Python packages will be picked up automatically.
Changes to Rust files will require a recompilation; see "Installing Qiskit from source" below.

You can easily install all the standard developer dependencies for in-place testing, documentation-building,
and linting using the `dev` dependency group:

```
pip install --group dev
```

### Set up a Conda environment

For Conda users, a new environment can be created as follows.

```
conda create -y -n QiskitDevenv python=3
conda activate QiskitDevenv
```

```
pip install -e .
```

## Installing Qiskit from source

Qiskit is primarily written in Python but there are some core routines
that are written in the [Rust](https://www.rust-lang.org/) programming
language to improve the runtime performance. For the released versions of
Qiskit we publish precompiled binaries on the
[Python Package Index](https://pypi.org/) for all the supported platforms
which only requires a functional Python environment to install. However, when
building and installing from source you will need a Rust compiler installed. You can do this very easily
using rustup: https://rustup.rs/ which provides a single tool to install and
configure the latest version of the rust compiler.
[Other installation methods](https://forge.rust-lang.org/infra/other-installation-methods.html)
exist too. For Windows users, besides rustup, you will also need to install
the Visual C++ build tools so that Rust can link against the system c/c++
libraries. You can see more details on this in the
[rustup documentation](https://rust-lang.github.io/rustup/installation/windows-msvc.html).

If you use Rustup, it will automatically install the correct Rust version
currently used by the project.

Once you have a Rust compiler installed, you can rely on the normal Python
build/install steps to install Qiskit. This means you run `pip install .` or
`pip install -e .` in your local git clone to build and install Qiskit.
Note that changes to Rust files will not be reflected in an editable install
until you recompile.

You can recompile the Rust components of an editable
install by running
```
python setup.py build_rust --inplace [--release | --debug]
```
Modifications to Rust files will not take effect until the Rust extension module
is recompiled with the above command.  You must have the "build" dependencies
installed in your environment for this command to work.  Do this by running
```
pip install --group build
```

By default, `pip install .` will build the Rust components in "release" mode
and `pip install -e .` (or `python setup.py build_rust --inplace`) will build
them in "debug" mode, without optimizations.  Debug mode will have poor
runtime performance.  You can set the environment variable `QISKIT_BUILD_PROFILE`
to `release` or `debug` to control the default.  The `--release`/`--debug` flag
to `build_rust` overrides this default.

### Compile time options

When building Qiskit from source there are options available to control how
Qiskit is built. These options are set with the following environment variables:

* `QISKIT_BUILD_WITH_MIMALLOC=1`: this will enable using
  [mimalloc](https://github.com/microsoft/mimalloc) as the global allocator for
  Qiskit instead of the default system allocator. This improves the runtime and
  memory performance of Qiskit but will require having a C compiler installed
  when building Qiskit.
* `QISKIT_NO_CACHE_GATES=1`: this will disable runtime caching of
  Python gate objects when accessing them from a `QuantumCircuit` or `DAGCircuit`.
  This makes a tradeoff between runtime performance for Python access and memory
  overhead. Caching gates will result in better runtime for users of Python at
  the cost of increased memory consumption. If you're working with any custom
  transpiler passes written in Python or are otherwise using a workflow that
  repeatedly accesses the `operation` attribute of a `CircuitInstruction` or `op`
  attribute of `DAGOpNode` enabling caching is recommended.

These environment variables are only valid when building Qiskit the Python package
with a PEP 517 compatible build tool or calling `setup.py` directly.
Or as a standalone C library with `make c` (`QISKIT_NO_CACHE_GATES` has no effect
when building a standalone C library).

## Issues and pull requests

We use [GitHub pull requests](https://help.github.com/articles/about-pull-requests) to accept
contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on, and if you
need help, you can reference the issue when discussing it with other community
and team members.

* For documentation issues relating to pages in the guides, tutorials, and migration guides sections of [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com/docs/), please open an issue in the [Qiskit/documentation repo](https://github.com/Qiskit/documentation/issues/new/choose) rather than the Qiskit/qiskit repo. In other words, any page that DOES NOT have `/api/` in the url should be addressed in the Qiskit/documentation repo.
* For issues relating to API reference pages (any page that contains `/api/` in the url), please open an issue in the repo specific to that API reference, for example [Qiskit/qiskit](https://github.com/Qiskit/qiskit/issues/new/choose), [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer/issues/new/choose), or [Qiskit/qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime/issues/new/choose).

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation, you can open a *Draft* pull request and prepend
the title with the **\[WIP\]** tag (for Work In Progress). This will indicate
to reviewers that the code in the PR isn't in its final state and will change.
It also means that we will not merge the commit until it is finished. You or a
reviewer can remove the [WIP] tag when the code is ready to be fully reviewed for merging.

Before marking your Pull Request as "ready for review" make sure you have followed the
PR Checklist below. PRs that adhere to this list are more likely to get reviewed and
merged in a timely manner.

### Pull request author checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the CI tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.

   If your code fails the local style checks (specifically the black
   or Rust code formatting check) you can use `tox -eblack` and
   `cargo fmt` to automatically fix the code formatting.

2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.

   If your pull request is adding a new class, function, or module that is
   intended to be user facing ensure that you've also added those to a
   documentation `autosummary` index to include it in the api documentation.

3. If you are of the opinion that the modifications you made warrant additional tests,
   feel free to include them.

4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a `reno` release note for that
   change and that the PR is tagged for the changelog.

5. All contributors have [signed the CLA](#contributor-licensing-agreement).

   You will need to ensure that all commits in the PR chain have a correctly configured
   email address, and the email address is registered to a GitHub account that has signed
   the CLA.  A bot will leave a comment with a link to sign the CLA.

6. The PR has a concise and explanatory title that can be understood without
   clicking on another GitHub issue.

   The PR title will become the summary line of the commit, which appears in `git log`.
   For example, "Fixes Issue 1234" is a bad title, and "Fix `ApplyLayout` with
   empty layouts" is good.

7. If the PR addresses an open issue the PR description includes the `fixes #issue-number`
   syntax to link the PR to that issue (**you must use the exact phrasing in order for GitHub
   to automatically close the issue when the PR merges**)

### Use of AI tools

> [!WARNING]
> If you use any AI tool while preparing your code contribution, you **must** disclose the name of the tool and its version in the PR description.

When using AI tools for code generation, your submission must still be your own original work of authorship, as required by the [Contributor License Agreement (CLA)](https://qisk.it/cla). It is your responsibility to make sure that:

- You review and fully understand the generated code, and you can explain the reasoning behind it during review.
- The usage of the AI tool does not violate any third-party license obligations.
- The AI tool's terms and conditions allow its output to be used in open source projects and are compatible with the [Qiskit license](LICENSE.txt), the [Qiskit CLA](https://qisk.it/cla), and [these Contributor Guidelines](CONTRIBUTING.md).
- You only use AI tools that have features to:
  * filter out generated code substantially similar to training data, or
  * identify similar training code so you can comply with the original license obligations (notice, attribution, etc.) and only contribute if it's compatible with the [Qiskit license](LICENSE.txt).
- You disclose the name and version of the AI tool in your PR description.

Submissions that appear unreviewed or copied directly from an AI tool without proper understanding may be requested to be revised or declined.

Remember that spamming issues or pull requests with AI-generated comments is prohibited under the [Qiskit Code of Conduct](https://qisk.it/coc).


### Code review

All code merged to Qiskit, even from maintainers, goes through a code-review
process after a pull request is made.  There are a small number of
maintainers who can authorize a final merge, but code review involves everyone
working together to make Qiskit better.  You can review code even if you
are not a maintainer, which helps make sure pull requests are technically
correct, well tested, and easier to tackle in their final maintainer review.

The code-review process is a normal part of software development, and nothing to
be scared of; for very easy changes it can be as simple as a maintainer saying
"looks good to me!" (or in short, "LGTM!") and merging the PR.  For more complex changes, it's often a
back-and-forth where the reviewer may ask a couple of questions about why things
were done a particular way, and make suggestions for improvement.  You don't
need to do everything suggested if you've got good reasons to disagree, but
communicate that clearly and politely.

If you're struggling with code review or a PR on Qiskit, you can ask for help in
the `#qiskit-pr-help` channel on [the public Qiskit Slack](https://qisk.it/join-slack).

Remember that the PR author is a human, not just a username!  It's OK to ask
questions about the code, but don't be mean or rude about it even if you don't
like it.  It's also fine to provide comments that are just compliments with no
suggested changes, if you particularly like something!

#### What to focus on in review

* Is everything in [the PR checklist](#pull-request-checklist) done?

* Are any new public APIs easy to use, well documented, and consistent with
  other parts of the Qiskit API?

* Do any changes to the code have knock-on effects for other parts of Qiskit
  that may be using them, or do they imply changes to the assumptions in
  our data structures?

* Are there any edge cases you can think of that the code might not handle well?
  Could the PR benefit from extra tests to cover these, or to verify other edge
  cases that it *does* handle successfully?

* Is the code reasonably easy for you to understand?  This particular point is
  tricky; the more you review code, the easier it will be for you to understand
  other code, so don't worry about this as much if you're getting started.

* If this PR is a bugfix, is it suitable for backport?  If not, could the PR be
  split into a "simple" bugfix that is suitable for backport and a follow-on
  improvement?  (Not all bugfixes *must* be backported.)

#### Writing review comments

* Make concrete suggestions when you think something should be changed, but
  remember that the author might have already thought about it and have a
  reason.  Try "What do you think about us raising a `TypeError` here instead of
  returning `None`?", rather than "You should raise an exception here".

* Try to make each round of review thorough.  Don't add one or two comments on
  one file, then come back a day later and add a couple of other unrelated
  comments on a different file, and so on.  Try to review the whole PR
  thoroughly in one go; it's easier to catch bugs like this, and less
  frustrating for the PR creator. If that's too much for you, consider
  asking if the PR could be split into smaller independent chunks.

* Try to keep the number of comments reasonable.  This depends on the size of
  the PR, but remember that there's somebody who'll read all your comments, and
  it can be demoralizing if you get a PR back and it's got 30 comments on from a
  50-line change.  If you feel like you're putting too many comments on,
  consider if you could group several of them into one theme, and ask them as a
  more detailed question with a focus on only one part of the code.  Try not to
  comment the same thing in many places.

* Try to avoid saying "you do" in review comments, and instead try to
  say things like "we do" even when talking about new code. It's not a big
  change, yet it helps to make us think about Qiskit's code as something that we
  all own and care about, and that we're all working together to make it better.

#### Responding to review comments

* Ask questions if you don't understand what a reviewer is saying, or if you're
  not certain whether they're suggesting changes.

* Feel free to respond to suggestions or questions with your reasoning for doing
  things a different way, if you don't fully agree with the review comment.
  Code review is a collaborative two-way process.

* Don't use the "update branch" on GitHub unless a maintainer suggests it or
  there are merge conflicts.  The merge queue will take care of this when the
  PR is approved, and pressing it unnecessarily uses up CI resources that other
  PRs might need.

* Try not to take suggestions personally.  It's hard to communicate over text,
  especially when we might have different native languages and we're talking
  about improving something.  Assume that the reviewer was acting in good faith,
  trying to be polite, and knows what they're talking about; it's unlikely that
  they meant to make you feel bad or insult you.  If you _do_ feel like somebody
  is not following [the code of conduct](/CODE_OF_CONDUCT.md), please report it
  using the violation form there.

#### Things that shouldn't be said

* Anything about the formatting of the code, unless it is illegible.  We have
  automated code formatters that enforce a uniform style, and CI requires them
  to have been run.

* Minor stylistic changes in _how_ people code, except where they might be
  seriously affecting performance or legibility.  There are lots of ways to
  program, especially in Python, and lots of ways that achieve the same thing.
  For example, if somebody has written
  ```python
  if my_condition:
      my_first_variable = 123
      my_second_variable = 456
  else:
      my_first_variable = 456
      my_second_variable = 123
  ```
  there's no need to suggest changes like
  ```python
  my_first_variable = 123 if my_condition else 456
  my_second_variable = 456 if my_condition else 123
  ```
  Both are perfectly legible, and focusing on small details like this is
  frustrating for everybody.

### Pull request merging checklist

When a PR is fully approved by code owners, it can be queued for merge.
Authorised users (those with write access to the repository) will be able to
press the "merge when ready" button.  Before enqueuing for merge, check that the
following PR metadata items are set correctly:

* The "milestone" is set to the expected release version.  For PRs to be
  backported, this should be (for example) "2.3.2".  For PRs for the next minor
  release, it should be (for example) "2.4.0".  If the PR is unrelated to any
  particular release (such as a change only to a test), you can leave this
  blank.

  This metadata lets us quickly jump from `git log` to the PR page, and see
  there which Qiskit release a patch went out in.

* The correct "Changelog: X" label is applied, including "Changelog: None" if
  the PR need not appear.

  These labels are much simpler than the `reno` structure; they are for the
  GitHub "releases" page instead, and categorize PRs into "Added", "Deprecated",
  "Changed" or "Fixed".

* Suitable backport commands have been set, if necessary.

  In most cases, applying the label "stable backport potential" is sufficient.
  In this case, the Mergify bot will open a backport PR to the most recent
  stable branch (for example `stable/2.3` if we are currently preparing for
  2.4.0).  If you need more complex backports, write a GitHub comment of the
  form:

  ```
  @Mergifyio backport <branch> <branch2> ...
  ```

  You can have as many branches as necessary.  It usually only necessary to do
  this to support old major branches.

* Any issues fixed by the PR have their own "Fix #<num>" line in the author's PR
  comment.  If you are empowered to merge PRs, you should be empowered to edit
  the author's comment to add these, if necessary.

* The PR title is clear, concise, and does not link to GitHub issues.

  As a merger, you can edit the title; there is an "edit" button at the top right
  of the PR main page, right of the title.  This title becomes the `git` commit
  summary line, so should be understandable without reference to GitHub.

If a PR is backported, the Mergify bot will open a PR for each branch to
backport it to.  Assuming there are no merge conflicts, you can immediately
approve and enqueue those PRs; a GitHub Actions workflow will copy across the
labels (except for "stable backport potential") and milestone.


## Contributor Licensing Agreement

Before you can submit any code, all contributors must sign a
contributor license agreement (CLA). By signing a CLA, you're attesting
that you are the author of the contribution, and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Qiskit project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](https://qisk.it/cla)
document is available for review as a PDF.

Note: If your contribution is part of your employment or your contribution
is the property of your employer, then you will more than likely need to sign a
[corporate CLA](https://qisk.it/corporate-cla) too and
email it to us at <qiskit@us.ibm.com>.

## Changelog generation

The changelog is automatically generated as part of the release process
automation. This works through a combination of the git log and the pull
request. When a release is tagged and pushed to github the release automation
bot looks at all commit messages from the git log for the release. It takes the
PR numbers from the git log (assuming a squash merge) and checks if that PR had
a `Changelog:` label on it. If there is a label it will add the git commit
message summary line from the git log for the release to the changelog.

If there are multiple `Changelog:` tags on a PR the git commit message summary
line from the git log will be used for each changelog category tagged.

The current categories for each label are configured in `qiskit_bot.yaml` in
the repository root.

## Release notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with:

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root:

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like:

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    :class:`.QuantumCircuit` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The :class:`.QuantumCircuit` class has a new method :meth:`~.QuantumCircuit.foo`.
    This is the equivalent of calling the :func:`~qiskit.foo` to do something to your
    :class:`.QuantumCircuit`. This is the equivalent of running :func:`~qiskit.foo`
    on your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    :func:`~qiskit.foo` function which provides similar functionality but with
    more accurate results and better performance. You should update your
    :func:`~qiskit.bar.foobar` calls to :func:`~qiskit.foo`.
```

You can also look at other release notes for other examples.

For the ``features``, ``deprecations``, and ``upgrade`` sections there are a
list of subsections available which are used to provide more structure to the
release notes organization. If you're adding a feature, making an API change,
or deprecating an API you should pick the subsection that matches that note.
For example if you're adding a new feature to the transpiler, you should put
it under the ``upgrade_transpiler`` section.

Note that you can use sphinx [restructured text syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
In fact, you can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

#### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/qiskit/issues/12345>` for more
    details.
```

#### Generating the release notes

After release notes have been added, you can use reno to see what the full output
of the release notes is. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full release
notes for all releases (since we started using reno during 0.9) you just run:

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged:

    reno report --version 0.9.0

#### Building release notes locally

Building The release notes are part of the standard qiskit documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

## Testing

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run Qiskit's Python test suite (as well as its Rust-based
unit tests if you've modified native code).

### Qiskit's Python test suite

The easiest way to run Qiskit's Python test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. Additionally, the
environment that tox sets up matches the CI environment more closely and it
runs the tests in parallel (resulting in much faster execution). To run tests
on all installed supported python versions and lint/style checks you can simply
run `tox`. Or if you just want to run the tests once run for a specific python
version: `tox -epy313` (or replace py313 with the python version you want to use).

If you just want to run a subset of tests you can pass a selection regex to
the test runner. For example, if you want to run all tests that have "dag" in
the test id you can run: `tox -epy310 -- dag`. You can pass arguments directly to
the test runner after the bare `--`. To see all the options on test selection
you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy310 -- -n test.python.compiler.test_transpiler
```
or to run the same module by path:

```
tox -epy310 -- -n test/python/compiler/test_transpiler.py
```
to run a class:

```
tox -epy310 -- -n test.python.compiler.test_transpiler.TestTranspile
```
to run a method:
```
tox -epy310 -- -n test.python.compiler.test_transpiler.TestTranspile.test_transpile_non_adjacent_layout
```

If you want to run the test suite in your local environment without using `tox` as a runner, you can
either use the `tox devenv -e py310` command to have `tox` construct you a new development environment,
or you can install the `test` dependency group using your package manager, such as
```
pip install --group test
```

Alternatively there is a makefile provided to run tests, however this
does not perform any environment setup. It also doesn't run tests in
parallel and doesn't provide an option to easily modify the tests run.
For executing the tests with the makefile, a `make test` target is available.
The execution of the tests (both via the make target and during manual
invocation) takes into account the `LOG_LEVEL` environment variable. If
present, a `.log` file will be created on the test directory with the
output of the log calls, which will also be printed to stdout. You can
adjust the verbosity via the content of that variable, for example:

Linux and Mac:

``` {.bash}
$ cd out
out$ LOG_LEVEL="DEBUG" ARGS="-V" make test
```

Windows:

``` {.bash}
$ cd out
C:\..\out> set LOG_LEVEL="DEBUG"
C:\..\out> set ARGS="-V"
C:\..\out> make test
```

For executing a simple python test manually, we don\'t need to change
the directory to `out`, just run this command:

Linux and Mac:

``` {.bash}
$ LOG_LEVEL=INFO python -m unittest test/python/circuit/test_circuit_operations.py
```

Windows:

``` {.bash}
C:\..\> set LOG_LEVEL="INFO"
C:\..\> python -m unittest test/python/circuit/test_circuit_operations.py
```

##### STDOUT/STDERR and logging capture

When running tests in parallel using `stestr` either via tox, the Makefile
(`make test_ci`), or in CI we set the env variable
`QISKIT_TEST_CAPTURE_STREAMS` which will capture any text written to stdout,
stderr, and log messages and add them as attachments to the tests run so
output can be associated with the test case it originated from. However, if
you run tests with `stestr` outside of these mechanisms by default the streams
are not captured. To enable stream capture just set the
`QISKIT_TEST_CAPTURE_STREAMS` env variable to `1`. If this environment
variable is set outside of running with `stestr` the streams (STDOUT, STDERR,
and logging) will still be captured but **not** displayed in the test runners
output. If you are using the stdlib unittest runner a similar result can be
accomplished by using the
[`--buffer`](https://docs.python.org/3/library/unittest.html#command-line-options)
option (e.g. `python -m unittest discover --buffer ./test/python`).

##### Test Skip Options

How and which tests are executed is controlled by an environment
variable, `QISKIT_TESTS`:

Option | Description | Default
------ | ----------- | -------
`run_slow` | It runs tests tagged as *slow*. | `False`

It is possible to provide more than one option separated with commas.

Alternatively, the `make test_ci` target can be used instead of
`make test` in order to run in a setup that replicates the configuration
we used in our CI systems more closely.

### Snapshot Testing for Visualizations

If you are working on code that makes changes to any matplotlib visualizations
you will need to check that your changes don't break any snapshot tests, and add
new tests where necessary. You can do this as follows:

1. Make sure you have pushed your latest changes to your remote branch.
2. Go to link: `https://mybinder.org/v2/gh/<github_user>/<repo>/<branch>?urlpath=apps/test/ipynb/mpl_tester.ipynb`. For example, if your GitHub username is `username`, your forked repo has the same name as the original, and your branch is `my_awesome_new_feature`, you should visit https://mybinder.org/v2/gh/username/qiskit/my_awesome_new_feature?urlpath=apps/test/ipynb/mpl_tester.ipynb.
This opens a Jupyter Notebook application running in the cloud that automatically runs
the snapshot tests (note this may take some time to finish loading).
3. Each test result provides a set of 3 images (left: reference image, middle: your test result, right: differences). In the list of tests the passed tests are collapsed and failed tests are expanded. If a test fails, you will see a situation like this:

   <img width="995" alt="Screenshot_2021-03-26_at_14 13 54" src="https://user-images.githubusercontent.com/23662430/112663508-d363e800-8e50-11eb-9478-6d665d0ff086.png">
4. Fix any broken tests. Working on code for one aspect of the visualizations
can sometimes result in minor changes elsewhere to spacing etc. In these cases
you just need to update the reference images as follows:
    - download the mismatched images (link at top of Jupyter Notebook output)
    - unzip the folder
    - copy and paste the new images into `qiskit/test/ipynb/mpl/references`,
  replacing the existing reference images
    - add, commit and push your changes, then restart the Jupyter Notebook app in your browser. The
  tests should now pass.
5. Add new snapshot tests covering your new features, extensions, or bugfixes.
    - add your new snapshot tests to `test/ipynb/mpl/test_circuit_matplotlib_drawer.py`
    , where you can also find existing tests to use as a guide.
    - commit and push your changes, restart the Jupyter Notebook app in your browser.
    As this is the first time you run your new tests there won't be any reference
    images to compare to. Instead you should see an option in the list of tests
    to download the new images, like so:

    <img width="1002" alt="Screenshot_2021-03-26_at_15 38 31" src="https://user-images.githubusercontent.com/23662430/112665215-b9c3a000-8e52-11eb-89e7-b18550718522.png">

    - download the new images, then copy and paste into `qiskit/test/ipynb/mpl/references`
    - add, commit and push your changes, restart the Jupyter Notebook app in your browser. The
    new tests should now pass.

Note: If you have run `test/ipynb/mpl_tester.ipynb` locally it is possible some file metadata has changed, **please do not commit and push changes to this file unless they were intentional**.


### Testing Rust components

Many of Qiskit's core data structures and algorithms are implemented in Rust.
The bulk of this code is exercised heavily by our Python-based unit testing,
but this coverage really only provides integration-level testing from the
perspective of Rust.

To provide Rust unit testing, we use `cargo test`. Rust tests are
integrated directly into the Rust file being tested within a `tests` module.
Functions decorated with `#[test]` within these modules are built and run
as tests.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn my_first_test() {
        assert_eq!(2, 1 + 1);
    }
}
```

For more detailed guidance on how to write Rust tests, you can refer to the Rust
documentation's [guide on writing tests](https://doc.rust-lang.org/book/ch11-01-writing-tests.html).

Rust tests are run separately from the Python tests. The easiest way to run
them is via `tox`, which creates an isolated venv and pre-installs `qiskit`
prior to running `cargo test`:

```bash
tox -erust
```

> [!TIP]
> If you've already built your changes (e.g. `python setup.py build_rust --release --inplace`),
> you can pass `--skip-pkg-install` when invoking `tox` to avoid a rebuild. This works because
> Python will instead find and use Qiskit from the current working directory (since we skipped
> its installation).

#### Using a custom venv instead of `tox`

If you're not using `tox`, you can also execute Cargo tests directly in your own virtual environment.
If you haven't done so already, [create a Python virtual environment](#set-up-a-python-venv) and
**_activate it_**.

You will need to install (at least) the `build` and `test` dependency groups, such as
```
pip install --group build --group test
```
You can alternatively install the `dev` group, which encompasses both of these.

Then, run the following commands:

```bash
python setup.py build_rust --inplace
tools/run_cargo_test.py
```

The first command builds Qiskit in editable mode,
which ensures that Rust tests that interact with Qiskit's Python code actually
use the latest Python code from your working directory. The second command invokes
the tests via Cargo.

#### Calling Python from Rust tests
By default, our Cargo project configuration allows Rust tests to interact with the
Python interpreter by calling `Python::with_gil` to obtain a `Python` (`py`) token.
This is particularly helpful when testing Rust code that (still) requires interaction
with Python.

To execute code that needs the GIL in your tests, define the `tests` module as
follows:

```rust
#[cfg(all(test, not(miri)))] // disable for Miri!
mod tests {
    use pyo3::prelude::*;
    
    #[test]
    fn my_first_test() {
        Python::with_gil(|py| {
            todo!() // do something that needs a `py` token.
        })
    }
}
```

> [!IMPORTANT]
> Note that we explicitly disable compilation of such tests when running with Miri, i.e.
`#[cfg(not(miri))]`. This is necessary because Miri doesn't support the FFI
> code used internally by PyO3.
>
> If not all of your tests will use the `Python` token, you can disable Miri on a per-test
basis within the same module by decorating *the specific test* with `#[cfg_attr(miri, ignore)]`
instead of disabling Miri for the entire module.


### Unsafe code and Miri

Any `unsafe` code added to the Rust logic should be exercised by Rust-space
tests, in addition to the more complete Python test suite.  In CI, we run the
Rust test suite under [Miri](https://github.com/rust-lang/miri) as an
undefined-behavior sanitizer.

Miri is currently only available on `nightly` Rust channels, so to run it
locally you will need to ensure you have that channel available, such as by
```bash
rustup install nightly --components miri
```

After this, you can run the Miri test suite with
```bash
MIRIFLAGS="<flags go here>" cargo +nightly miri test
```

For the current set of `MIRIFLAGS` used by Qiskit's CI, see the
[`miri.yml`](https://github.com/Qiskit/qiskit/blob/main/.github/workflows/miri.yml)
GitHub Action file.  This same file may also include patches to dependencies to
make them compatible with Miri, which you would need to temporarily apply as
well.

### Testing the C API

The C API test suite is located at `test/c/`.  This is a CMake project and uses
CMake's `ctest` runner.  To build and run the tests, use the `ctest` recipe in
the top-level `Makefile`, which you can run with

```bash
make ctest
```

#### Writing C API tests

The C API test suite automatically discovers any files inside `test/c/` matching
the pattern `test_*.c`. Each one of these files should follow a template similar
to the following.
```c
#include "common.h"

// Individual tests may be implemented by custom functions. The return value
// should be `Ok` (from `test/c/common.h`) when the test was successful or one
// of the other error codes (`>0`) indicating the error type.
//
// Individual test functions should be marked static; this is a double line of
// defence so the compiler will error if you forget to add it to the runner.
static int test_something()
{
    return Ok;
}

// One main function must exist, WHOSE FUNCTION NAME MATCHES THE FILENAME!
int test_FILE_NAME()
{
    // Ideally, this function should track the number of failed subtests.
    int num_failed = 0;

    // The RUN_TEST macro will execute the provided test function and perform a
    // minimal amount of logging to indicate the success/failure of this test.
    num_failed += RUN_TEST(test_something);

    // Finally, this test should report the number of failed subtests.
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);

    // And return the number of failed subtests. If this is greater than 0,
    // ctest will indicate the failure.
    return num_failed;
}
```

## Style and lint

Qiskit uses three tools for Python code formatting and lint checking. The
first tool is [black](https://github.com/psf/black) which is a code formatting
tool that will automatically update the code formatting to a consistent style.
The second tool is [ruff](https://docs.astral.sh/ruff/) which is a code linter
which does a deeper analysis of the Python code to find both style issues and
potential bugs and other common issues in Python.

You can check that your local modifications conform to the style rules by
running `tox -elint` which will run `black` and  `ruff` to check the
local code formatting and lint. If black returns a code formatting error you can
run `tox -eblack` to automatically update the code formatting to conform to the
style. However, if `ruff` returns any error you will have to fix these issues by
manually updating your code. Sometimes `ruff` will be able to fix failures with
the `--fix` flag. In these cases the output will tell you how many errors can be
automatically fixed.

Because they are so fast, it is sometimes convenient to run the tools `black` and `ruff` separately
rather than via `tox`.  You can install all the lint dependencies using the `lint` or `dev`
dependency groups, such as by
```
pip install --group lint
```
After this, `ruff` and `black` will be available and can be run from the command line. See
[`tox.ini`](tox.ini) for how `tox` invokes them.

### Rust style and lint

For formatting and lint checking Rust code, you'll need to use different tools than you would for Python. Qiskit uses [rustfmt](https://github.com/rust-lang/rustfmt) for
code formatting. You can simply run `cargo fmt` (if you installed Rust with the
default settings using `rustup`), and it will update the code formatting automatically to
conform to the style guidelines. This is very similar to running `tox -eblack` for Python code. For lint checking, Qiskit uses [clippy](https://github.com/rust-lang/rust-clippy) which can be invoked via `cargo clippy`.

Rust lint and formatting checks are included in the `tox -elint` command. For CI to pass you will need both checks to pass without any warnings or errors. Note that this command checks the code but won't apply any modifications, if you need to update formatting, you'll need to run `cargo fmt`.

### C style and lint

Qiskit uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format C code.
The style is based on LLVM, with a few Qiskit-specific adjustments.
To check whether the C code conforms to the style guide, you can run `make cformat`. This check
will need to execute without any warnings or errors for CI to pass.
Automatic formatting can be applied by `make fix_cformat`.

## Building API docs locally

The API documentation is built with Sphinx.
We recommend that you use [**tox**](https://tox.readthedocs.io/en/latest) to orchestrate this.
Run a complete documentation build with
```
tox -e docs
```

The documentation output will be located at `docs/_build/html`.
Open the `index.html` file there in your browser to find the main page.

To build the documentation you will need to have Doxygen installed and in
your PATH environment variable as tox will run `doxygen` to build the API
documentation for the C API. You can download doxygen from [here](https://www.doxygen.nl/download.html).

### Troubleshooting docs builds

When you build documentation, you might get errors that look like
```
ValueError: earliest-version set to unknown revision '1.0.0rc1'
```
If so, you need to fetch Qiskit's `git` tags and stable branches, in order to fully build the release notes.
To do this, run the command:
```
git fetch --tags upstream
```
where `upstream` is your name for the [git remote repository](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) that corresponds to https://github.com/Qiskit/qiskit (this repository).
You might need to re-run this command if Qiskit has issued a new release since the last time you built the documentation.

Sometimes, you might get errors about "names not existing" or "failed to import" during the docs build, even when the test suite passes.
This can mean that Sphinx's cache has become invalidated, but hasn't been successfully cleared.
Use the command:
```
tox -e docs-clean
```
to fully clean out all documentation build artefacts and partial builds, and see if the problem persists.


## Development cycle

The development cycle for qiskit is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes.

### Branches

* `main`:

The main branch is used for development of the next version of qiskit.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.8 contains the code for the
0.8.2 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

In the lead up to a release there are a few things to keep in mind. Prior to
the release date there is a feature, removal, and deprecation proposal freeze
date. This date in each release cycle is the last day where a new PR adding a
new feature, removing something, or adding a new deprecation can be proposed (in
a ready for review state) for potential inclusion in the release. If a new
PR is opened after this date it will not be considered for inclusion in that
release. Note, that meeting these deadlines does not guarantee inclusion in a
release: they are preconditions. You can refer to the milestone page for each
release to see these dates for each release (for example for 0.21.0 the page is:
https://github.com/Qiskit/qiskit/milestone/23).

After the proposal freeze a release review period will begin, during this time
release candidate PRs will be reviewed as we finalize the feature set and merge
the last PRs for the release. Following the review period a release candidate will be
tagged and published. This release candidate is a pre-release that enables users and
developers to test the release ahead of time. When the pre-release is tagged the release
automation will publish the pre-release to PyPI (but only get installed on user request),
create the `stable/*` branch, and generate a pre-release changelog/release page. At
this point the `main` opens up for development of the next release. The `stable/*`
branches should only receive changes in the form of bug fixes at this point. If there
is a need additional release candidates can be published from `stable/*` and when the
release is ready a full release will be tagged and published from `stable/*`.

## Adding deprecation warnings
The Qiskit code is part of Qiskit and, therefore, the [Qiskit Deprecation Policy](./DEPRECATION.md) fully applies here. Additionally, Qiskit's testsuite breaks if a `DeprecationWarning` is emitted. If you are deprecating code, you should add a test to use the new/non-deprecated method (most of the time based on the existing test of the deprecated method) and alter the existing test to check that the deprecated method still works as expected, [using `assertWarns`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertWarns). The `assertWarns` context will silence the deprecation warning while checking that it raises.

For example, if `Obj.method1` is being deprecated in favour of `Obj.method2`, the existing test (or tests) for `method1` might look like this:

```python
def test_method1(self):
   result = Obj.method1()
   self.assertEqual(result, <expected>)
```

Deprecating `method1` means that `Obj.method1()` now raises a deprecation warning and the test will not pass. The existing test should be updated and a new test added for `method2`:


```python
def test_method1_deprecated(self):
   with self.assertWarns(DeprecationWarning):
       result = Obj.method1()
   self.assertEqual(result, <expected>)

def test_method2(self):
   result = Obj.method2()
   self.assertEqual(result, <expected>)
```

`test_method1_deprecated` can be removed after `Obj.method1` is removed (following the [Qiskit Deprecation Policy](./DEPRECATION.md)).

## Using dependencies

We distinguish between "requirements" and "optional dependencies" in Qiskit.
A requirement is a package that is absolutely necessary for core functionality in Qiskit, such as NumPy or SciPy.
An optional dependency is a package that is used for specialized functionality, which might not be needed by all users.
If a new feature has a new dependency, it is almost certainly optional.


### Version support policy

For Python-space dependencies, Qiskit follows [the scientific-computing standard SPEC 0](https://scientific-python.org/specs/spec-0000/).
In short: Qiskit will require versions of NumPy and SciPy that are at least two years old.
For packages not covered by SPEC 0, the requirements must be satisfiable with published binary artifacts from PyPI for all supported Python versions on [all platforms with tier 1 and tier 2 support](https://quantum.cloud.ibm.com/docs/guides/install-qiskit#operating-system-support).

Python dependencies that are optional at runtime, only used during the build, or only used during the development process are not constrained.
Qiskit supports all versions of CPython that are not end of life, which is wider than the minimum SPEC 0 support.

Rust dependencies are not constrained, other than by the platform support requirements and minimum supported Rust version of the repository.

### Adding a requirement

Any new requirement must have broad system support; it needs to be supported on all the Python versions and operating systems that qiskit supports.
It also cannot impose many version restrictions on other packages.
Users often install qiskit into virtual environments with many different packages in, and we need to ensure that neither we, nor any of our requirements, conflict with their other packages.
When adding a new requirement, you must add it to [`requirements.txt`](requirements.txt) following the [version-support policy](#version-support-policy).

### Adding an optional dependency

New features can also use optional dependencies, which might be used only in very limited parts of qiskit.
These are not required to use the rest of the package, and so should not be added to `requirements.txt`.
Instead, if several optional dependencies are grouped together to provide one feature, you can consider adding an "extra" to the package metadata, such as the `visualization` extra that installs Matplotlib and Seaborn (amongst others).
To do this, modify the [`setup.py`](setup.py) file, adding another entry in the `extras_require` keyword argument to `setup()` at the bottom of the file.
You should also add a new "tester" to [`qiskit.utils.optionals`](qiskit/utils/optionals.py), for use in the next section.

### Checking for optionals

You cannot `import` an optional dependency at the top of a file, because if it is not installed, it will raise an error and qiskit will be unusable.
We also largely want to avoid importing packages until they are actually used; if we import a lot of packages during `import qiskit`, it becomes sluggish for the user if they have a large environment.
Instead, you should use [one of the "lazy testers" for optional dependencies](https://quantum.cloud.ibm.com/docs/api/qiskit/utils#optional-dependency-checkers), and import your optional dependency inside the function or class that uses it, as in the examples within that link.
Very lightweight _requirements_ can be imported at the tops of files, but even this should be limited; it's always ok to `import numpy`, but Scipy modules are relatively heavy, so only import them within functions that use them.


## Dealing with the git blame ignore list

In the qiskit repository we maintain a list of commits for git blame
to ignore. This is mostly commits that are code style changes that don't
change the functionality but just change the code formatting (for example,
when we migrated to use black for code formatting). This file,
`.git-blame-ignore-revs` just contains a list of commit SHA1s you can tell git
to ignore when using the `git blame` command. This can be done one time
with something like

```
git blame --ignore-revs-file .git-blame-ignore-revs qiskit/version.py

```

from the root of the repository. If you'd like to enable this by default you
can update your local repository's configuration with:

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

which will update your local repository's configuration to use the ignore list
by default.
