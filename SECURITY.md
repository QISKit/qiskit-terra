# Security Policy

## Safe Usage Policy

> [!IMPORTANT]
> Nothing in this section, nor any Qiskit API documentation, should be taken to imply any legal guarantee.
> These safety policies are not a waiver of the "AS IS" basis of the software with respect to warranty and liability set out in sections 7 and 8 of Qiskit's Apache-2.0 license.

This section is the general policy of the library, and may be superseded by individual function or
module documentation.  For example, the deserialization functions for QPY, OpenQASM 2 and OpenQASM 3
all have specific security policies that modify those listed here.

In all cases, unless otherwise noted:

* Qiskit makes no guarantees how much memory or runtime any given function
  call will take. All users should be prepared to recover from process memory exhaustion or
  excessive runtime.  In particular, memory exhaustion _may_ cause process termination, regardless
  of further security guarantees.
* Interfaces marked "experimental" provide no security guarantees.
* Interfaces that have marked "safety" contracts provide no security guarantees if the caller does
  not fulfil the specified contract (for example, setting the `check=False` argument to
  `SparseObservable.from_raw_parts` has no security guarantees on untrusted input).
* Functions that are not part of the public interface have no security guarantees.

If these criteria are met, the programming intention is that, for any user input:

* Qiskit will not read out-of-bounds memory, including via stack overflows.
* Calling Qiskit functions will not abort the process.  This rule does not yet apply to the C API.

If you find a violation of this contract, please *do not* open a public bug report, but instead
follow the vulnerability process described below.

The following are best-effort intentions (except where overridden by specific policies).  Violations
of these, subject to the same preconditions as the guarantees, are considered non-critical bugs and
should be opened on the public issue tracker:

* Qiskit will not cause Rust-space panics and will not raise `PanicException` to Python space.
* Qiskit will not leak unbounded amounts of memory.

## Supported Versions

Qiskit supports the most recent major release with new features, which will only appear in minor releases of that series.
The most recent minor release in the current major release series is also supported with bug fixes.
In addition, the last minor release of the *previous* major release series is supported with bug fixes for six months after a new major release,
and one year of security fixes.

For example, if the most recent release is 1.0.1, then the current major release series is 1.x and the current minor release is 1.0.x.
The 1.0.x series will be supported with bug fixes, until the release of 1.1.0, which will include new features.
The last version of the previous major release, 0.46.x, is supported with bug fixes until six months after the final release of 1.0.0,
and for one year with any security fixes.

We provide more detail on [the release and support schedule of Qiskit in our documentation](https://quantum.cloud.ibm.com/docs/open-source/qiskit-sdk-version-strategy).

## Remediation Process

The repository's remediation process is explained in [the "security" section of the maintainers' guide](/MAINTAINING.md#security).

## Reporting a Vulnerability

To report vulnerabilities, you can privately report a potential security issue
via the GitHub security vulnerabilities feature. This can be done here:

https://github.com/Qiskit/qiskit/security/advisories

Please do **not** open a public issue about a potential security vulnerability.

You can find more details on the security vulnerability feature in the GitHub
documentation here:

https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability
