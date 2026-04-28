# DEPRECATED: use utils.image.regions.core_types instead
import warnings
warnings.warn(
    "utils.image.regions.core_image is deprecated; import from utils.image.regions.core_types (or utils.image.regions) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core_types import *  # re-export
