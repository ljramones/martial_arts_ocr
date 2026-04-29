from __future__ import annotations

from .doclayout_yolo import DocLayoutYOLOStrategy


class GenericYOLOLayoutStrategy(DocLayoutYOLOStrategy):
    """Optional generic YOLO adapter for local experiments.

    Generic YOLO is useful only when pointed at a locally trained document
    region model. It is not a default document-layout solution.
    """

    name = "generic_yolo"
