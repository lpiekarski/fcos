import os
import subprocess

import torch
from PIL.Image import Image
from torch import nn
import torchvision
import modules
import box_coders
import yaml


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.load(f, yaml.CLoader)


def initialize_from_config(cfg):
    locals()
    return exec(f"{cfg[0]}({', '.join(cfg[1])})")


def get_image_and_label_paths(file):
    with open(file, "r") as f:
        image_paths = f.readlines()
        label_paths = [os.path.normpath(os.path.join(os.path.dirname(img), '..', 'labels', os.path.splitext(os.path.basename(img))[0] + '.txt')) for img in image_paths]
        return image_paths, label_paths


def get_inputs_and_targets(data, device):
    if "setup" in data:
        subprocess.call(data["setup"], shell=True)

    train_file = data["train"]
    image_paths, label_paths = get_image_and_label_paths(train_file)

    img_to_tensor = torchvision.transforms.ToTensor()
    inputs = [img_to_tensor(Image.open(img)) for img in image_paths]
    targets = []
    for input_img, label_path in inputs, label_paths:
        img_w = input_img.shape[-1]
        img_h = input_img.shape[-2]
        with open(label_path, 'r') as f:
            bboxes = [line.split() for line in f.readlines()]
            cls = [bb[0] for bb in bboxes]
            ctr_x = [img_w * bb[1] for bb in bboxes]
            ctr_y = [img_h * bb[2] for bb in bboxes]
            w = [img_w * bb[3] for bb in bboxes]
            h = [img_h * bb[4] for bb in bboxes]
            targets.append({
                "labels": torch.tensor(cls, dtype=torch.int64, device=device),
                "boxes": torch.tensor([[cx - pw / 2., cy - ph / 2., cx + pw / 2., cy + ph / 2.] for cx, cy, pw, ph in zip(ctr_x, ctr_y, w, h)], dtype=torch.float32, device=device)
            })
    return inputs, targets
