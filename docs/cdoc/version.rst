.. _capi-version:

==========
Versioning
==========

Qiskit's version can be queried using a set of compiler macros.

.. c:macro:: QISKIT_VERSION

    A human-readable version as ``char*``. For example: ``"2.1.0-rc1"``.

.. c:macro:: QISKIT_VERSION_HEX

    The version number as 4-byte hexadecimal. The format is ``0xMMmmppls``, where
    ``M`` is the major, ``m`` is the minor, ``p`` is the patch, ``l`` is the release level
    and ``s`` is the serial. For example, 2.1.0-rc1 is ``0x020100C1``.

    Note that final versions, such as ``2.1.0``, will always end with ``0xF0`` in the last byte.

    See :c:func:`qk_api_version` for the runtime version of this.

.. c:macro:: QISKIT_VERSION_MAJOR

    The major release version.

.. c:macro:: QISKIT_VERSION_MINOR

    The minor release version.

.. c:macro:: QISKIT_VERSION_PATCH

    The patch release version.

.. c:macro:: QISKIT_RELEASE_LEVEL

    The release level:

    * ``0xA`` for an unreleased dev (or alpha) version
    * ``0xB`` for a beta version
    * ``0xC`` for a release candidate
    * ``0xF`` for a stable (or final) version.

.. c:macro:: QISKIT_RELEASE_SERIAL

    This can be used to indicate the pre-release number in a pre-release series.
    For example, this would be set to ``4`` for ``2.1.0-rc4``. This is ``0`` for the final version.

.. c:macro:: QISKIT_GET_VERSION_HEX(major, minor, patch, level, serial)

    A macro to pack the version numbers into hexadecimal format. This can be used as
    tool to compare numbers, for example to ensure the current version is at least the
    stable 2.1.0 release do:

    .. code-block:: c

        if (QISKIT_VERSION_HEX >= QISKIT_GET_VERSION_HEX(2, 1, 0, 0xF, 0)) {
            // Code for version 2.1.0 (final) or later
        }

.. doxygenfunction:: qk_api_version
