# Security Policy

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
