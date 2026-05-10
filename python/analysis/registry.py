"""
registry.py — analysis registration and lookup.

Each analysis is a function that takes a BDTrajectory and returns a dict of
named NumPy arrays / scalars. The @register_analysis decorator records a
descriptor for the framework so that:

  - missing required datasets are caught before the analysis runs,
  - cache invalidation can compare an `inputs_hash` of the kwargs,
  - the reduce step knows the per-trajectory output shapes,
  - users can list available analyses by name.

An optional `combine` callable can be attached to the descriptor via
`@analysis.combine` (set on the AnalysisDescriptor returned by the decorator)
or by passing combine=... directly to register_analysis.

Note for users: when an external package registers analyses, the
@code_git_sha attribute the framework writes into the cache only reflects the
in-repo analysis code. Provenance for external packages is opt-in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# In-process registry. Keyed by analysis name.
_REGISTRY: dict[str, "AnalysisDescriptor"] = {}


@dataclass
class AnalysisDescriptor:
    """Everything the framework needs to know about a registered analysis."""

    name: str
    version: int
    requires: tuple[str, ...]
    outputs: dict[str, dict[str, Any]]
    analyze: Callable[..., dict[str, Any]]
    combine: Optional[Callable[[list[dict[str, Any]]], dict[str, Any]]] = None
    doc: str = ""

    # Implementation note: kwargs hashing is delegated to inputs_hash() so the
    # cache.py module produces identical hashes whether the call is happening
    # in a worker (kwargs known) or the reduce step (kwargs read back from the
    # cache attrs).

    def inputs_hash(self, kwargs: dict[str, Any]) -> str:
        """8-char SHA1 of (analysis name, version, sorted kwargs)."""
        payload = json.dumps(
            {"name": self.name, "version": self.version, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

    def __call__(self, traj, **kwargs) -> dict[str, Any]:
        return self.analyze(traj, **kwargs)


def register_analysis(
    *,
    name: str,
    version: int,
    requires: tuple[str, ...] | list[str] = (),
    outputs: dict[str, dict[str, Any]],
    combine: Optional[Callable[[list[dict[str, Any]]], dict[str, Any]]] = None,
):
    """
    Decorator: register `func` as the analyze() callable for analysis `name`.

    Returns the AnalysisDescriptor (not the raw function) so callers can
    optionally attach a combine() via descriptor.combine = ... or call
    descriptor(traj) like a function.
    """

    if not name or not name.replace("_", "").isalnum():
        raise ValueError(
            f"Analysis name {name!r} must be non-empty alphanumeric/underscore."
        )

    def _wrap(func: Callable[..., dict[str, Any]]) -> AnalysisDescriptor:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            # Allow re-registration only if the function object is the same —
            # otherwise treat as a programmer error (typo, copy-paste).
            if existing.analyze is not func:
                raise ValueError(
                    f"Analysis {name!r} is already registered "
                    f"(existing version={existing.version}, "
                    f"new version={version})."
                )
            return existing

        desc = AnalysisDescriptor(
            name=name,
            version=int(version),
            requires=tuple(requires),
            outputs=dict(outputs),
            analyze=func,
            combine=combine,
            doc=(func.__doc__ or "").strip(),
        )
        _REGISTRY[name] = desc
        return desc

    return _wrap


def get_analysis(name: str) -> AnalysisDescriptor:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown analysis {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_analyses() -> list[str]:
    return sorted(_REGISTRY)


def resolve_analysis_names(spec: str | list[str]) -> list[str]:
    """Resolve --analyses CLI input ('all' | comma-list | list[str]) to names."""
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    if spec == ["all"]:
        return list_analyses()
    unknown = [n for n in spec if n not in _REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown analyses: {unknown}. Registered: {list_analyses()}"
        )
    return list(spec)
