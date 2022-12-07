from typing import Dict, List, Tuple
from torchvision.ops import boxes as box_ops
import torch
from torch import Tensor


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