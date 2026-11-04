# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for providers."""
from __future__ import annotations

import logging
from collections.abc import Callable

from qiskit.providers.backend import Backend

logger = logging.getLogger(__name__)


def filter_backends(
    backends: list[Backend], filters: Callable[[Backend], bool] | None = None, **kwargs
) -> list[Backend]:
    """Return the backends matching the specified filtering.

    Filter the `backends` list by their `configuration` or `status`
    attributes, or from a boolean callable. The criteria for filtering can
    be specified via `**kwargs` or as a callable via `filters`, and the
    backends must fulfill all specified conditions.

    Args:
        backends (list[Backend]): list of backends.
        filters (callable): filtering conditions as a callable.
        **kwargs: dict of criteria.

    Returns:
        list[Backend]: a list of backend instances matching the
            conditions.
    """

    def _match_all(obj, criteria):
        """Return True if all items in criteria matches items in obj."""
        return all(getattr(obj, key_, None) == value_ for key_, value_ in criteria.items())

    # Inspect the backends to decide which filters belong to
    # backend.configuration and which ones to backend.status, as it does
    # not involve querying the API.
    configuration_filters = {}
    status_filters = {}
    for key, value in kwargs.items():
        if all(key in backend.configuration() for backend in backends):
            configuration_filters[key] = value
        else:
            status_filters[key] = value

    # 1. Apply backend.configuration filtering.
    if configuration_filters:
        backends = [b for b in backends if _match_all(b.configuration(), configuration_filters)]

    # 2. Apply backend.status filtering (it involves one API call for
    # each backend).
    if status_filters:
        backends = [b for b in backends if _match_all(b.status(), status_filters)]

    # 3. Apply acceptor filter.
    backends = list(filter(filters, backends))

    return backends


def resolve_backend_name(
    name: str, backends: list[Backend], deprecated: dict[str, str], aliased: dict[str, list[str]]
) -> str:
    """Resolve backend name from a deprecated name or an alias.

    A group will be resolved in order of member priorities, depending on
    availability.

    Args:
        name (str): name of backend to resolve
        backends (list[Backend]): list of available backends.
        deprecated (dict[str: str]): dict of deprecated names.
        aliased (dict[str: list[str]]): dict of aliased names.

    Returns:
        str: resolved name (name of an available backend)

    Raises:
        LookupError: if name cannot be resolved through regular available
            names, nor deprecated, nor alias names.
    """
    # account for BackendV2
    available = []
    for backend in backends:
        available.append(backend.name() if backend.version == 1 else backend.name)

    resolved_name = deprecated.get(name, aliased.get(name, name))
    if isinstance(resolved_name, list):
        resolved_name = next((b for b in resolved_name if b in available), "")

    if resolved_name not in available:
        raise LookupError(f"backend '{name}' not found.")

    if name in deprecated:
        logger.warning("Backend '%s' is deprecated. Use '%s'.", name, resolved_name)

    return resolved_name
