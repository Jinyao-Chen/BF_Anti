"""Compatibility shim: the tracker now lives in perception/bbox_tracker.py.

The real implementation moved so the onboard runtime and the Gazebo simulator
share one tracker without importing the sim package. Existing sim scripts that
import BboxTracker from the sim directory keep working.
"""
from perception.bbox_tracker import BBox, BboxTracker, TrackOut

__all__ = ["BboxTracker", "TrackOut", "BBox"]
