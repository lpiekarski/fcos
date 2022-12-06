import argparse

import torchvision
from PIL import Image
import subprocess
import modules
import box_coders
from torch import nn
import torch.optim as optim
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from tqdm import tqdm
import random

from losses import FCOSLoss
from modules import FCOS
from utils import load_yaml_config, initialize_from_config, get_image_and_label_paths, get_inputs_and_targets
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FCOS_train",
        description="Train the fcos model",
    )
    parser.add_argument('--data', default='configurations/datasets/coco.yaml')
    parser.add_argument('--hyp', default='configurations/hyperparameters/hyp.fcos.yaml')
    parser.add_argument('--weights')
    args = parser.parse_args()

    data = load_yaml_config(args.data)
    hyp = load_yaml_config(args.hyp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anchor_sizes = tuple([(x,) for x in hyp["strides"]])  # equal to strides of multi-level feature map
    aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one "anchor" per location
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    strides = hyp["strides"]
    out_channels = hyp["out_channels"]
    backbone = hyp["backbone"]
    backbone = exec(f"{backbone[0]}({', '.join([str(v) for v in backbone[1]])})")
    box_coder = hyp["box_coder"]
    box_coder = exec(f"{box_coder[0]}({', '.join([str(v) for v in box_coder[1]])})")
    num_classes = data["num_classes"]
    convs_in_heads = hyp["convs_in_heads"]
    detections_per_img = hyp["detections_per_img"]
    warmup_lr = hyp["warmup_lr"]
    base_lr = hyp["lr"]
    momentum = hyp["momentum"]
    epochs = hyp["epochs"]
    inputs, targets = get_inputs_and_targets(data, device)
    image_mean = [img.flatten() for img in inputs]
    image_mean = torch.stack([img for img in image_mean])
    image_mean, image_std = image_mean.mean(), torch.std(image_mean)
    transform = hyp["transform"]
    transform = exec(f"{transform[0]}({', '.join([str(v) for v in transform[1]])})")
    fcos = FCOS(
        backbone=backbone,
        box_coder=box_coder,
        transform=transform,
        num_classes=num_classes,
        num_convs_in_heads=convs_in_heads,
        anchor_generator=anchor_generator,
        detections_per_img=detections_per_img
    )
    fcos.to(device)

    num_parameters = sum(p.numel() for p in fcos.parameters() if p.requires_grad)
    print(f"{fcos}")
    print(f"Device: {device}")
    print(f"Number of trainable parameters: {num_parameters}")

    optimizer = optim.SGD(fcos.parameters(), base_lr, momentum=momentum)
    loss_function = FCOSLoss()

    epoch_cls = []
    epoch_bbox = []
    epoch_ctrness = []
    epoch_total = []
    for epoch in range(epochs):
        fcos.train()
        cls_losses = []
        bbox_losses = []
        ctrness_losses = []
        sum_losses = []

        if len(warmup_lr) > 0:
            lr = warmup_lr.pop(0)
        else:
            lr = base_lr
        optimizer.param_groups[0]['lr'] = lr

        with tqdm(total=len(inputs)) as pbar:
            idxs = list(range(len(inputs)))
            random.shuffle(idxs)
            for idx in idxs:
                optimizer.zero_grad()
                outputs = fcos(inputs[idx], targets[idx])
                losses = loss_function(*outputs)
                classification = losses["classification"]
                bbox_regression = losses["bbox_regression"]
                bbox_ctrness = losses["bbox_ctrness"]
                loss = classification + bbox_regression + bbox_ctrness
                loss.backward()
                optimizer.step()
                cls_losses.append(classification.item())
                bbox_losses.append(bbox_regression.item())
                ctrness_losses.append(bbox_ctrness.item())
                sum_losses.append(loss.item())
                pbar.set_description(f"classification: {np.mean(cls_losses):.5f}, bbox regression: {np.mean(bbox_losses):.5f}, centerness: {np.mean(ctrness_losses):.5f}, total: {np.mean(sum_losses):.5f}")
                pbar.update(1)
            fcos.eval()
            with torch.no_grad():
                pbar.set_description(f"Epoch {epoch + 1:02d}/{epochs:02d} classification: {np.mean(cls_losses):.5f}, bbox regression: {np.mean(bbox_losses):.5f}, centerness: {np.mean(ctrness_losses):.5f}, total: {np.mean(sum_losses):.5f}")
                epoch_bbox.append(np.mean(bbox_losses))
                epoch_ctrness.append(np.mean(ctrness_losses))
                epoch_cls.append(np.mean(cls_losses))
                epoch_total.append(np.mean(sum_losses))
