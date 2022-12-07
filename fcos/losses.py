from torch import nn
import torch
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss


class FCOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = ClassificationLoss()
        self.ctrness_loss = CenternessLoss()
        self.bbox_reg_loss = BBoxRegressionLoss()

    def forward(self, targets, head_outputs, anchors, num_anchors_per_level):
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
        return self.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
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

        loss_cls = self.cls_loss(cls_logits, all_gt_classes_targets, foregroud_mask)
        loss_bbox_reg = self.bbox_reg_loss(anchors, bbox_regression, all_gt_boxes_targets, foregroud_mask)
        loss_bbox_ctrness = self.ctrness_loss(anchors, bbox_ctrness, all_gt_boxes_targets, foregroud_mask)

        return {
            "classification": loss_cls,
            "bbox_regression": loss_bbox_reg,
            "bbox_ctrness": loss_bbox_ctrness,
        }


class CenternessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, bbox_ctrness, all_gt_boxes_targets, foregroud_mask):
        num_foreground = foregroud_mask.sum().item()
        bbox_ctrness = bbox_ctrness[foregroud_mask]
        anchors = torch.stack(anchors)
        anchors = anchors[foregroud_mask]
        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        all_gt_boxes_targets = all_gt_boxes_targets[foregroud_mask]
        reg_targets = self.box_coder.encode_single(anchors, all_gt_boxes_targets)
        l, t, r, b = reg_targets[:, 0], reg_targets[:, 1], reg_targets[:, 2], reg_targets[:, 3]
        ctrness_targets = torch.sqrt((torch.min(l, r) / torch.max(l, r)) * (torch.min(t, b) / torch.max(t, b)))
        return sigmoid_focal_loss(bbox_ctrness.squeeze(), ctrness_targets, reduction="sum") / max(1, num_foreground)


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cls_logits, all_gt_classes_targets, foregroud_mask):
        num_foreground = foregroud_mask.sum().item()
        one_hot = torch.zeros_like(cls_logits, device=cls_logits.device)
        one_hot[foregroud_mask] = nn.functional.one_hot(all_gt_classes_targets[foregroud_mask], num_classes=cls_logits.shape[-1]).type(torch.float32)
        return sigmoid_focal_loss(cls_logits, one_hot, reduction="sum") / max(1, num_foreground)


class BBoxRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors, bbox_regression, all_gt_boxes_targets, foregroud_mask):
        num_foreground = foregroud_mask.sum().item()
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
        return generalized_box_iou_loss(bbox_regression - offset, all_gt_boxes_targets - offset, reduction="sum") / max(1, num_foreground)