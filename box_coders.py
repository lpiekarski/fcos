from torch import Tensor
import torch


class BoxCoder:
    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        raise NotImplementedError()

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        raise NotImplementedError()


class BoxLinearCoder(BoxCoder):
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size: bool = True) -> None:
        """
        Args:
            normalize_by_size (bool): normalize deltas by the size of src (anchor) boxes.
        """
        self.normalize_by_size = normalize_by_size

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded

        Returns:
            Tensor: the encoded relative box offsets that can be used to
            decode the boxes.
        """
        # get the center of reference_boxes
        reference_boxes_ctr_x = 0.5 * (reference_boxes[:, 0] + reference_boxes[:, 2])
        reference_boxes_ctr_y = 0.5 * (reference_boxes[:, 1] + reference_boxes[:, 3])

        # get box regression transformation deltas
        target_l = reference_boxes_ctr_x - proposals[:, 0]
        target_t = reference_boxes_ctr_y - proposals[:, 1]
        target_r = proposals[:, 2] - reference_boxes_ctr_x
        target_b = proposals[:, 3] - reference_boxes_ctr_y

        targets = torch.stack((target_l, target_t, target_r, target_b), dim=1)
        if self.normalize_by_size:
            reference_boxes_w = reference_boxes[:, 2] - reference_boxes[:, 0]
            reference_boxes_h = reference_boxes[:, 3] - reference_boxes[:, 1]
            reference_boxes_size = torch.stack(
                (reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h), dim=1
            )
            targets = targets / reference_boxes_size

        return targets

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.

        Returns:
            Tensor: the predicted boxes with the encoded relative box offsets.
        """

        boxes = boxes.to(rel_codes.dtype)

        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        if self.normalize_by_size:
            boxes_w = boxes[:, 2] - boxes[:, 0]
            boxes_h = boxes[:, 3] - boxes[:, 1]
            boxes_size = torch.stack((boxes_w, boxes_h, boxes_w, boxes_h), dim=1)
            rel_codes = rel_codes * boxes_size

        pred_boxes1 = ctr_x - rel_codes[:, 0]
        pred_boxes2 = ctr_y - rel_codes[:, 1]
        pred_boxes3 = ctr_x + rel_codes[:, 2]
        pred_boxes4 = ctr_y + rel_codes[:, 3]
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
        return pred_boxes