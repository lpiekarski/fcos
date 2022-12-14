from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn, Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from box_coders import BoxCoder
from fcos.postprocessor import postprocess_detections


class Backbone(nn.Module):
    def __init__(self, strides=None):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32]
        self.first_block = nn.Sequential(
            Conv(1, strides[0], (3, 3), padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [nn.Sequential(*[
                Conv(strides[i - 1], strides[i], (3, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ]) for i in range(1, len(strides))
             ]
        )

    def forward(self, x: Tensor) -> List[Tensor]:
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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        nn.init.constant_(self.conv.bias, 1)
        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.conv = Conv(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.relu = nn.ReLU()
        nn.init.constant_(self.conv.bias, 1)
        nn.init.normal_(self.conv.weight, std=0.01)

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
        self.conv_blocks = nn.Sequential(*[ConvBlock(in_channels) for _ in range(num_convs)])
        self.conv = Conv(
            in_channels,
            num_classes,
            kernel_size=3,
            padding=1,
            stride=1
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        aux = [self.conv_blocks(layer) for layer in x]  # aux: [(N, C, S, S) for stride S]
        aux = [self.conv(layer) for layer in aux]  # aux: [(N, C, S, S) for stride S]
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
        self.conv_blocks = nn.Sequential(*[ConvBlock(in_channels) for _ in range(num_convs)])
        self.bbox_head = Conv(in_channels, 4, kernel_size=3, padding=1, stride=1)
        self.ctrness_head = Conv(in_channels, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = [self.conv_blocks(layer) for layer in x]  # x: [(N, in_channels, S, S) for stride S]
        bbox_regression = [self.bbox_head(layer) for layer in x]  # bbox_regression: [(N, 4, S, S) for stride S]
        bbox_regression = [nn.functional.relu(layer) for layer in bbox_regression]  # bbox_regression: [(N, 4, S, S) for stride S]
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
            return targets, head_outputs, anchors, num_anchors_per_level
        else:
            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            return detections
