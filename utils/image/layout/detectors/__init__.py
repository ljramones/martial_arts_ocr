# utils/image/layout/detectors/__init__.py
from typing import List, Optional
import numpy as np
from utils.image.regions.core_image import ImageRegion

class BaseDetector:
    def detect(self, gray: np.ndarray) -> List[ImageRegion]:
        raise NotImplementedError
