# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Any, List

import torch

from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead, mask_rcnn_inference
from detectron2.projects.point_rend import ImplicitPointRendMaskHead
from detectron2.projects.point_rend.point_features import point_sample
from detectron2.projects.point_rend.point_head import roi_mask_point_loss
from detectron2.structures import Instances

from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage

from .point_utils import get_point_coords_from_point_annotation, compute_pairwise_term

__all__ = [
    "ImplicitPointRendLidarSupHead",
    "MaskRCNNConvUpsampleLidarSupHead",
]


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleLidarSupHead(MaskRCNNConvUpsampleHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.

    The difference with `MaskRCNNConvUpsampleHead` is that this head is trained
    with point supervision. Please use the `MaskRCNNConvUpsampleHead` if you want
    to train the model with mask supervision.
    """
    # def __init__(self) -> None:
    #     self.pairwise_size, self.pairwise_dilation, self.pairwise_color_thresh = 3, 2, 0.3


    def forward(self, x, instances: List[Instances], similaritiy_feature: torch.Tensor) -> Any:
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            N, C, H, W = x.shape
            assert H == W

            proposal_boxes = [x.proposal_boxes for x in instances]
            assert N == np.sum(len(x) for x in proposal_boxes)

            if N == 0:
                return {"loss_mask": x.sum() * 0}

            # Training with point supervision
            # Sanity check: annotation should not contain gt_masks
            assert not instances[0].has("gt_masks")
            point_coords, point_labels = get_point_coords_from_point_annotation(instances)

            mask_logits = point_sample(
                x,
                point_coords,
                align_corners=False,
            )

            #Training with pairwise_loss
            # assert similaritiy_feature != None
            # gt_classes = []
            # for instances_per_image in instances:
            #     if len(instances_per_image) == 0:
            #             continue
            #     gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            #     gt_classes.append(gt_classes_per_image)

            # gt_classes = torch.cat(gt_classes, dim=0)
            # total_num_masks = x.size(0)
            # indices = torch.arange(total_num_masks)
            # pairwise_mask_logits = x[indices, gt_classes]
            # pairwise_mask_logits = torch.unsqueeze(pairwise_mask_logits, dim=1) 

            # self.pairwise_size, self.pairwise_dilation, self.pairwise_color_thresh = 3, 2, 0.3


            # pairwise_losses = compute_pairwise_term(pairwise_mask_logits, self.pairwise_size, self.pairwise_dilation)
            # weights = (similaritiy_feature >= self.pairwise_color_thresh).float()

            # storage = get_event_storage()
            # warmup_factor = min(storage.iter / float(1000), 1.0)

            # loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            # loss_pairwise = loss_pairwise * warmup_factor

            return {"loss_mask": roi_mask_point_loss(mask_logits, instances, point_labels),
                    #"loss_pairwise": loss_pairwise
                    }
        else:
            mask_rcnn_inference(x, instances)
            return instances


@ROI_MASK_HEAD_REGISTRY.register()
class ImplicitPointRendLidarSupHead(ImplicitPointRendMaskHead):
    def _uniform_sample_train_points(self, instances):
        assert self.training
        assert not instances[0].has("gt_masks")
        point_coords, point_labels = get_point_coords_from_point_annotation(instances)

        return point_coords, point_labels
