"""LoveDA dataset for remote sensing semantic segmentation."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LoveDADataset(Dataset):
    """LoveDA dataset with 7 classes + ignore index.

    Label encoding:
        0=Background, 1=Building, 2=Road, 3=Water,
        4=Barren, 5=Forest, 6=Agriculture, 7=Ignore
    """

    CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
    NUM_CLASSES = 7
    IGNORE_INDEX = 7

    _NORM_MAP = {
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "sat493m": ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143]),
    }

    def __init__(self, root: str, split: str = "Train", transform: bool = True,
                 image_size: int = 512, input_norm: str = "imagenet"):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        if input_norm not in self._NORM_MAP:
            raise ValueError(f"Unknown input_norm={input_norm}. Choose from {list(self._NORM_MAP)}")
        self.input_norm = input_norm

        self.images: list[str] = []
        self.masks: list[str] = []

        for scene in ["Urban", "Rural"]:
            img_dir = os.path.join(root, split, scene, "images_png")
            mask_dir = os.path.join(root, split, scene, "masks_png")
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".png"):
                    self.images.append(os.path.join(img_dir, fname))
                    self.masks.append(os.path.join(mask_dir, fname))

        print(f"LoveDA {split}: {len(self.images)} images")

        mean, std = self._NORM_MAP[self.input_norm]
        self.normalize = transforms.Normalize(mean=mean, std=std)
        print(f"LoveDA input_norm: {self.input_norm}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform and self.split == "Train":
            image, mask = self._train_augment(image, mask)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()

        # Map ignore label to 255 for PyTorch CrossEntropyLoss(ignore_index=255)
        mask[mask == self.IGNORE_INDEX] = 255

        return image, mask

    def _train_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        sz = self.image_size

        # Random crop
        if h >= sz and w >= sz:
            top = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
            image = image[top : top + sz, left : left + sz]
            mask = mask[top : top + sz, left : left + sz]
        else:
            image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        # Random vertical flip
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

        # Random resize (scale 0.5–2.0) then crop back to sz
        # Use current image size (after crop) for correct scaling
        cur_h, cur_w = image.shape[:2]
        if random.random() > 0.5:
            scale = random.uniform(0.5, 2.0)
            new_h, new_w = int(cur_h * scale), int(cur_w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h >= sz and new_w >= sz:
                top = random.randint(0, new_h - sz)
                left = random.randint(0, new_w - sz)
                image = image[top : top + sz, left : left + sz]
                mask = mask[top : top + sz, left : left + sz]
            else:
                image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        return image, mask
