from importlib.util import find_spec
import warnings

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
    PhotoMetricDistortionMultiViewImage,
    PadMultiViewImage,
    NormalizeMultiviewImage,
    CustomCollect3D,
)
from .models.opt.adamw import AdamW2
from .models.utils import *
from .bevformer import *

_DD3D_OPTIONAL_DEPS = ('detectron2', 'fvcore', 'nuscenes', 'pyquaternion',
                       'seaborn')
_MISSING_DD3D_DEPS = [dep for dep in _DD3D_OPTIONAL_DEPS
                      if find_spec(dep) is None]

if _MISSING_DD3D_DEPS:
    # DD3D depends on additional third-party packages (for example detectron2).
    # Keep the top-level plugin import working when those extras are absent so
    # the standard BEVFormer training entrypoints can still start.
    warnings.warn(
        'Skipping DD3D plugin imports because optional dependencies are not '
        f'available: {", ".join(_MISSING_DD3D_DEPS)}. Install the DD3D extras '
        'to use BEVFormer V2 configs.',
        RuntimeWarning,
    )
else:
    from .dd3d import *
