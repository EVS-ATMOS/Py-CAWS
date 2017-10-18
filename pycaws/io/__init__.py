"""
================
Input and output
================

Functions to read, process and write stage and weather data.

    stage_output

"""

from .stage_processing import stage_output

__all__ = [s for s in dir() if not s.startswith('_')]
