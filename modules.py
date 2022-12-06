from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from torchvision.ops import boxes as box_ops
import torch
from torch import nn, Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from box_coders import BoxCoder


class Backbone(nn.Module):
    def __init__(self, strides=None):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]
        self.first_block = nn.Sequential(
            nn.Conv2d(1, strides[0], (3, 3), padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [nn.Sequential(*[
                nn.Conv2d(strides[i - 1], strides[i], (3, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ]) for i in range(1, len(strides))
             ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        aux = [x]
        for block in self.blocks:
            x = block(aux[-1])
            aux.append(x)
        return aux


class BackboneWithFPN(nn.Module):
    def __init__(self, strides, out_channels=32) -> None:
        super().__init__()
        self.strides = strides
        self.out_channels = out_channels
        self.backbone = Backbone(self.strides)
        self.fpn = FeaturePyramidNetwork(self.strides, self.out_channels)

    def forward(self, x: Tensor):
        output_backbone = self.backbone(x)
        x = OrderedDict()
        for i, f in enumerate(output_backbone):
            x[f'feat{i}'] = f
        output_fpn = self.fpn(x)
        return output_fpn


class Conv(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class FCOSClassificationHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_convs: int = 4,
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(*[Conv(in_channels) for _ in range(num_convs)])
        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=3,
            padding=1,
            stride=1
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        aux = [self.conv(self.conv_blocks(layer)) for layer in x]  # aux: [(N, C, S, S) for stride S]
        aux = [layer.transpose(2, 3) for layer in aux]  # aux: [(N, C, S, S) for stride S]
        aux = [layer.reshape(*layer.shape[:2], -1) for layer in aux]  # aux: [(N, C, S * S) for stride S]
        aux = torch.cat(aux, dim=2)  # aux: (N, C, A)
        aux = aux.transpose(1, 2)  # aux: (N, A, C)
        return aux


class FCOSRegressionHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_convs: int = 4,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[Conv(in_channels) for _ in range(num_convs)])
        self.bbox_head = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, stride=1)
        self.ctrness_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = [self.conv_blocks(layer) for layer in x]  # x: [(N, in_channels, S, S) for stride S]
        bbox_regression = [nn.functional.relu(self.bbox_head(layer)) for layer in x]  # bbox_regression: [(N, 4, S, S) for stride S]
        ctrness_regression = [self.ctrness_head(layer) for layer in x]  # ctrness_regression: [(N, 1, S, S) for stride S]

        bbox_regression = [layer.transpose(2, 3) for layer in bbox_regression]  # bbox_regression: [(N, 4, S, S) for stride S]
        ctrness_regression = [layer.transpose(2, 3) for layer in ctrness_regression]  # ctrness_regression: [(N, 1, S, S) for stride S]

        bbox_regression = [layer.reshape(*layer.shape[:2], -1) for layer in bbox_regression]  # bbox_regression: [(N, 4, S * S) for stride S]
        ctrness_regression = [layer.reshape(*layer.shape[:2], -1) for layer in ctrness_regression]  # ctrness_regression [(N, 1, S * S) for stride S]

        bbox_regression = torch.cat(bbox_regression, dim=2)  # bbox_regression: (N, 4, A)
        ctrness_regression = torch.cat(ctrness_regression, dim=2)  # ctrness_regression: (N, 1, A)

        bbox_regression = bbox_regression.transpose(1, 2)  # bbox_regression: (N, A, 4)
        ctrness_regression = ctrness_regression.transpose(1, 2)  # ctrness_regression: (N, A, 1)

        return bbox_regression, ctrness_regression


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, box_coder: BoxCoder, num_convs: Optional[int] = 4) -> None:
        super().__init__()
        self.box_coder = box_coder
        self.classification_head = FCOSClassificationHead(in_channels, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_convs)

    def compute_loss(
            self,
            targets: List[Dict[str, Tensor]],
            head_outputs: Dict[str, Tensor],
            anchors: List[Tensor],  # anchors/locations
            matched_idxs: List[Tensor],  # tells to which bounding box anchors are matched, -1 mean no matches
    ) -> Dict[str, Tensor]:
        cls_logits = head_outputs["cls_logits"]  # [N, A, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, A, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, A, 1]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []

        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
            gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_classes_targets = torch.stack(all_gt_classes_targets)

        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        loss_cls = self.compute_loss_cls(cls_logits, all_gt_classes_targets, foregroud_mask)
        loss_bbox_reg = self.compute_loss_bbox_reg(anchors, bbox_regression, all_gt_boxes_targets, foregroud_mask)
        loss_bbox_ctrness = self.compute_loss_ctrness(anchors, bbox_ctrness, all_gt_boxes_targets, foregroud_mask)

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }

    def compute_loss_ctrness(self, anchors, bbox_ctrness, all_gt_boxes_targets, foregroud_mask):
        bbox_ctrness = bbox_ctrness[foregroud_mask]
        anchors = torch.stack(anchors)
        anchors = anchors[foregroud_mask]
        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        all_gt_boxes_targets = all_gt_boxes_targets[foregroud_mask]
        reg_targets = self.box_coder.encode_single(anchors, all_gt_boxes_targets)
        l, t, r, b = reg_targets[:, 0], reg_targets[:, 1], reg_targets[:, 2], reg_targets[:, 3]
        ctrness_targets = torch.sqrt((torch.min(l, r) / torch.max(l, r)) * (torch.min(t, b) / torch.max(t, b)))
        return sigmoid_focal_loss(bbox_ctrness.squeeze(), ctrness_targets, reduction="sum")

    def compute_loss_bbox_reg(self, anchors, bbox_regression, all_gt_boxes_targets, foregroud_mask):
        anchors = torch.stack(anchors)
        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        anchors = anchors[foregroud_mask]
        bbox_regression = bbox_regression[foregroud_mask]
        all_gt_boxes_targets = all_gt_boxes_targets[foregroud_mask]
        bbox_regression = self.box_coder.decode_single(bbox_regression, anchors)
        # Move boxes so that 0 <= x0 < x1 and 0 <= y0 < y1 is always satisfied (required by generalized_box_iou_loss)
        min_vals = bbox_regression.min(dim=0).values
        min_x = min_vals[[0, 2]].min().clip(max=0)
        min_y = min_vals[[1, 3]].min().clip(max=0)
        offset = torch.tensor([min_x, min_y, min_x, min_y], device=bbox_regression.device)
        return generalized_box_iou_loss(bbox_regression - offset, all_gt_boxes_targets - offset, reduction="sum")

    def compute_loss_cls(self, cls_logits, all_gt_classes_targets, foregroud_mask):
        cls_logits = cls_logits[foregroud_mask]
        all_gt_classes_targets = all_gt_classes_targets[foregroud_mask]
        one_hot = nn.functional.one_hot(all_gt_classes_targets, num_classes=cls_logits.shape[-1]).type(torch.float32)
        return sigmoid_focal_loss(cls_logits, one_hot, reduction="sum")

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        }


class FCOS(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            box_coder: BoxCoder,
            num_classes: int,
            transform,
            # Anchor parameters
            anchor_generator: AnchorGenerator = None,
            center_sampling_radius: float = 1.5,
            score_thresh: float = 0.2,
            nms_thresh: float = 0.6,
            detections_per_img: int = 100,
            topk_candidates: int = 1000,
            num_convs_in_heads: int = 4,
            **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.head = FCOSHead(backbone.out_channels, num_classes, num_convs=num_convs_in_heads, box_coder=box_coder)
        self.box_coder = box_coder
        self.transform = transform

        self.center_sampling_radius = center_sampling_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

    def compute_loss(
            self,
            targets: List[Dict[str, Tensor]],
            head_outputs: Dict[str, Tensor],
            anchors: List[Tensor],
            num_anchors_per_level: List[int],
    ) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):  # batch
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]  # Match anchors
            # center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1]:] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)
        # matched index - anchor-to-target match
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(
            self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]],
            image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
                    box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                max_logits = logits_per_level.max(dim=-1)
                scores_per_level = torch.sqrt(
                    torch.sigmoid(max_logits.values) * torch.sigmoid(box_ctrness_per_level.squeeze(dim=1)))
                keep_idxs = scores_per_level > self.score_thresh

                box_regression_per_level = box_regression_per_level[keep_idxs]
                anchors_per_level = anchors_per_level[keep_idxs]
                scores_per_level = scores_per_level[keep_idxs]

                if self.topk_candidates < scores_per_level.shape[0]:
                    topk_idxs = torch.topk(scores_per_level, self.topk_candidates).indices
                else:
                    topk_idxs = torch.sort(scores_per_level, descending=True).indices

                scores_per_level = scores_per_level[topk_idxs]

                topk_idxs = max_logits.indices[keep_idxs][topk_idxs] + topk_idxs * num_classes
                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections

    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        # transform the input (normalise with std and )
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # compute the fcos heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        if self.training:
            losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
            return losses
        else:
            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            return detections
