"""
analyses — built-in analyses, registered on import.

Importing this subpackage from analysis/__init__.py is what populates the
registry. Adding a new analysis is just creating a new module here, decorating
your function with @register_analysis, and adding the import below.
"""

from . import msd as _msd  # noqa: F401
from . import force_orientation as _fo  # noqa: F401
from . import hexatic as _hexatic  # noqa: F401
from . import clusters as _clusters  # noqa: F401
from . import correlations as _correlations  # noqa: F401
from . import contact_duration as _contact_duration  # noqa: F401
