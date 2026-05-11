"""
analysis — cluster-side processing pipeline for GPUParticles parameter sweeps.

Public API:
    PsweepCollector  — read-only navigator over a sweep directory.
    ProcessedSweep   — reader for per-analysis `<name>.h5` reduced outputs.
    register_analysis — decorator that registers an analysis with the framework.
    get_analysis, list_analyses — registry lookup.
"""

from __future__ import annotations

import os as _os

# Set HDF5 file locking off before h5py is imported anywhere in the package.
# Defends against ghost-lock issues on Lustre/NFS shared filesystems and is
# harmless on local disks. See plan: caveat #2.
_os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from .registry import register_analysis, get_analysis, list_analyses  # noqa: E402
from .collector import PsweepCollector  # noqa: E402
from .processed import ProcessedSweep  # noqa: E402

# Importing the analyses subpackage triggers @register_analysis side effects so
# the registry is fully populated by the time anyone uses get_analysis().
from . import analyses as _analyses  # noqa: F401, E402

__all__ = [
    "PsweepCollector",
    "ProcessedSweep",
    "register_analysis",
    "get_analysis",
    "list_analyses",
]
