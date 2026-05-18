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

try:
    from .dd3d import *
except ModuleNotFoundError as exc:
    # DD3D depends on additional third-party packages (for example detectron2).
    # Keep the top-level plugin import working when those extras are absent so
    # the standard BEVFormer training entrypoints can still start.
    if exc.name is None or exc.name.startswith('projects.mmdet3d_plugin.dd3d'):
        raise
    warnings.warn(
        f'Skipping DD3D plugin imports because optional dependency "{exc.name}" '
        'is not installed. Install the DD3D extras to use bevformerv2 configs.',
        RuntimeWarning,
    )
