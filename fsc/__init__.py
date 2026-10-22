"""Faithful Seam Carving: content-aware resizing that does not silently delete or fuse objects."""

from .engine import resize, CarveResult, VERTICAL, HORIZONTAL  # noqa: F401
from .integrity import InstanceIntegrity  # noqa: F401

__version__ = "0.1.0"
