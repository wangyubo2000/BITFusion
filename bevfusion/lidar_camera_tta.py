# ---------------------------------------------------------------
# custom_3d_tta.py : Test-Time Augmentation for 3D detection (LiDAR / BEVFusion)
# ---------------------------------------------------------------

from typing import List
from mmengine.model import BaseTTAModel
from mmdet3d.registry import MODELS
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d


@MODELS.register_module()
class Det3DTTAModel(BaseTTAModel):
    """Test-time augmentation model for LiDAR or LiDAR+Camera BEVFusion."""

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_preds(self, aug_results: List[List[dict]]):
        """Merge predictions from multiple augmented inputs.

        Args:
            aug_results (List[List[dict]]): Outer list = batch,
                inner list = predictions under different augmentations.

        Returns:
            List[dict]: Merged 3D detection results per sample.
        """
        merged_results = []
        for results_per_sample in aug_results:
            # Each results_per_sample is a list of dicts:
            # [{'bbox_3d': BaseInstance3DBoxes, 'scores_3d': Tensor, 'labels_3d': Tensor}, ...]
            merged_results.append(
                merge_aug_bboxes_3d(
                    aug_results=results_per_sample,
                    aug_batch_input_metas=[
                        res['metainfo'] if 'metainfo' in res else {} for res in results_per_sample
                    ],
                    test_cfg=self.tta_cfg))
        return merged_results
