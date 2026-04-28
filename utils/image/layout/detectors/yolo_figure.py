from __future__ import annotations
import importlib.util
from typing import List, Dict, Any
import cv2, numpy as np
from utils.image.regions.core_image import ImageRegion
from . import BaseDetector


class YOLOFigureDetector(BaseDetector):
    available = importlib.util.find_spec("ultralytics") is not None

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed; YOLO figure detection is unavailable")
        else:
            self._yolo_cls = YOLO
        self.model = YOLO(cfg["yolo_model_path"])
        self.conf  = float(cfg.get("yolo_conf", 0.22))
        self.iou   = float(cfg.get("yolo_iou", 0.60))
        self.imgsz = int(cfg.get("yolo_imgsz", 1536))
        self.tta   = bool(cfg.get("yolo_tta", False))

    def detect(self, image: np.ndarray) -> List[ImageRegion]:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        result = self.model.predict(source=bgr, imgsz=self.imgsz,
                                    conf=self.conf, iou=self.iou,
                                    augment=self.tta, verbose=False)[0]
        out = []
        if result.boxes is None: return out
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy()):
            out.append(ImageRegion(
                x=int(x1), y=int(y1),
                width=int(x2-x1), height=int(y2-y1),
                region_type="figure", confidence=float(conf)
            ))
        return out
