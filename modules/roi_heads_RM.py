# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads


# def m_subsample_labels(
#         labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
# ):
#     """
#     Return `num_samples` (or fewer, if not enough found)
#     random samples from `labels` which is a mixture of positives & negatives.
#     It will try to return as many positives as possible without
#     exceeding `positive_fraction * num_samples`, and then try to
#     fill the remaining slots with negatives.
#
#     Args:
#         labels (Tensor): (N, ) label vector with values:
#             * -1: ignore
#             * bg_label: background ("negative") class
#             * otherwise: one or more foreground ("positive") classes
#         num_samples (int): The total number of labels with value >= 0 to return.
#             Values that are not sampled will be filled with -1 (ignore).
#         positive_fraction (float): The number of subsampled labels with values > 0
#             is `min(num_positives, int(positive_fraction * num_samples))`. The number
#             of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
#             In order words, if there are not enough positives, the sample is filled with
#             negatives. If there are also not enough negatives, then as many elements are
#             sampled as is possible.
#         bg_label (int): label index of background ("negative") class.
#
#     Returns:
#         pos_idx, neg_idx (Tensor):
#             1D vector of indices. The total length of both is `num_samples` or fewer.
#     """
#     positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
#     negative = nonzero_tuple(labels == bg_label)[0]
#     # num_pos = int(num_samples * positive_fraction)
#     # protect against not enough positive examples
#     # num_pos = min(positive.numel(), num_pos)
#     # num_neg = num_samples - num_pos
#     # protect against not enough negative examples
#     # num_neg = min(negative.numel(), num_neg)
#     num_pos = positive.numel()
#     num_neg = negative.numel()
#     print(positive)
#     print(negative)
#     # randomly select positive and negative examples
#     perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
#     perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
#
#     pos_idx = positive[perm1]
#     neg_idx = negative[perm2]
#     return pos_idx, neg_idx

@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_RM(StandardROIHeads):

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        # sampled_fg_idxs, sampled_bg_idxs = m_subsample_labels(
        #     gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        # )
        # sampled_idxs = torch.cat([sampled_bg_idxs, sampled_fg_idxs], dim=0)
        sampled_idxs = torch.arange(gt_classes.size(0)).to(gt_classes.device)
        return sampled_idxs, gt_classes

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        nongt_dims = [proposal.proposal_boxes.tensor.size(0) for proposal in proposals]

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        if self.training:
            losses = self._forward_box(features, proposals, nongt_dims)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, nongt_dims)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
            self, features: Dict[str, torch.Tensor], proposals: List[Instances], nongt_dims: List[int]
    ) -> Union[Dict[str, torch.Tensor], List[Instances], List[int]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        # add geometric feature
        pred_boxes = [x.proposal_boxes.tensor for x in proposals]
        # embedding relation module*********************************************************
        box_features = self.box_head(box_features, pred_boxes, nongt_dims)
        # ********************************************************************
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
