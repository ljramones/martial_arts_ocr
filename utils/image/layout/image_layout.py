"""
Legacy shim for backward compatibility.

All logic has moved to:
    utils.image.layout.analyzer.LayoutAnalyzer

This file exists so that any older imports such as:
    from utils.image.image_layout import LayoutAnalyzer
still work without modification.
"""

from utils.image.layout.analyzer import LayoutAnalyzer

__all__ = ["LayoutAnalyzer"]
