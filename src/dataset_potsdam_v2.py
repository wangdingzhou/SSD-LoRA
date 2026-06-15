"""Potsdam (ISPRS) dataset for remote sensing semantic segmentation.

Directory structure:
    root/
    ├── 2_Ortho_RGB/          — 38 tiles, 6000x6000 RGB TIF
    │   ├── top_potsdam_2_10_RGB.tif
    │   └── ...
    └── *.tif                  — 38 label TIFs (color-coded)
        ├── top_potsdam_2_10_label.tif
        └── ...

Classes (5 + ignore):
    0=Impervious, 1=Building, 2=LowVeg, 3=Tree, 4=Car
    255=Clutter (ignore)

Standard split (ADVMSeg): 18 train / 6 val from 24 annotated tiles
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PotsdamDataset(Dataset):
    """Potsdam dataset with 5 classes + ignore.

    Label color map (BGR as read by cv2):
        (255, 255, 255) -> 0  Impervious
        (0,   0,   255) -> 1  Building (Blue in BGR)
        (255, 255, 0  ) -> 2  LowVeg (Cyan in BGR: B=255, G=255, R=0)
        (0,   255, 0  ) -> 3  Tree (Green)
        (0,   255, 255) -> 4  Car (Yellow in BGR: B=0, G=255, R=255)
        (0,   0,   0  ) -> 255  Clutter (Black, but actually Red=255,0,0 in RGB)
        (255, 0,   0  ) -> 255  Clutter (Red in BGR -> Blue in RGB? No: BGR(0,0,255)=Red)
    """

    # BGR color -> class index (verified from actual label TIF files)
    COLOR_MAP = {
        (255, 255, 255): 0,   # Impervious (White) — 20.6%
        (255, 0,   0  ): 1,   # Building (Blue in RGB) — 15.1%
        (255, 255, 0  ): 2,   # Low vegetation (Cyan in RGB) — 42.2%
        (0,   255, 0  ): 3,   # Tree (Green) — 7.4%
        (0,   255, 255): 4,   # Car (Yellow in RGB) — 0.9%
        (0,   0,   255): 255, # Clutter (Red in RGB) — 13.7%
    }

    CLASS_NAMES = ["Impervious", "Building", "LowVeg", "Tree", "Car"]
    NUM_CLASSES = 5
    IGNORE_INDEX = 255

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    # Standard ADVMSeg split: 18 train / 6 val from 24 annotated tiles
    # Tiles numbered: 2_10..14, 3_10..14, 4_10..15, 5_10..15, 6_7..15, 7_7..13
    # Val tiles (commonly used): 2_12, 4_12, 5_15, 6_8, 7_7, 7_10
    DEFAULT_TRAIN_TILES = [
        "2_10", "2_11", "2_13", "2_14",
        "3_10", "3_11", "3_12", "3_13", "3_14",
        "4_10", "4_11", "4_13", "4_14", "4_15",
        "5_10", "5_11", "5_12", "5_14",
    ]
    DEFAULT_VAL_TILES = [
        "2_12", "4_12", "5_15", "6_8", "7_7", "7_10",
    ]

    def __init__(self, root: str, split: str = "train", transform: bool = True,
                 image_size: int = 512, tile_ids: list = None,
                 samples_per_tile: int = 1):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.samples_per_tile = samples_per_tile if (split == "train" and transform) else 1

        # Determine tile split
        if tile_ids is not None:
            self.tile_ids = tile_ids
        elif split == "train":
            self.tile_ids = self.DEFAULT_TRAIN_TILES
        else:
            self.tile_ids = self.DEFAULT_VAL_TILES

        img_dir = os.path.join(root, "2_Ortho_RGB")

        self.images: list[str] = []
        self.masks: list[str] = []

        for tid in self.tile_ids:
            img_name = f"top_potsdam_{tid}_RGB.tif"
            mask_name = f"top_potsdam_{tid}_label.tif"
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(root, mask_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

        self.num_tiles = len(self.images)
        print(f"Potsdam {split}: {self.num_tiles} tiles x {self.samples_per_tile} samples = {len(self)} total")

        self.normalize = transforms.Normalize(mean=self._MEAN, std=self._STD)

    def _color_to_label(self, mask_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR label image to class index map using nearest-color matching."""
        h, w = mask_bgr.shape[:2]
        colors = np.array(list(self.COLOR_MAP.keys()), dtype=np.int16)
        indices = np.array(list(self.COLOR_MAP.values()), dtype=np.uint8)
        pixels = mask_bgr.reshape(-1, 3).astype(np.int16)
        dists = ((pixels[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
        nearest = dists.argmin(axis=1)
        label = indices[nearest].reshape(h, w)
        return label

    def __len__(self) -> int:
        return self.num_tiles * self.samples_per_tile

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_idx = idx % self.num_tiles
        image = cv2.imread(self.images[tile_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_bgr = cv2.imread(self.masks[tile_idx])
        mask = self._color_to_label(mask_bgr)

        if self.transform and self.split == "train":
            image, mask = self._train_augment(image, mask)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def _train_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        sz = self.image_size

        # Random crop from 6000x6000
        if h >= sz and w >= sz:
            top = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
        else:
            top, left = 0, 0
        image = image[top:top + sz, left:left + sz]
        mask = mask[top:top + sz, left:left + sz]

        # Random flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random scale
        if random.random() > 0.5:
            scale = random.uniform(0.5, 2.0)
            new_h, new_w = int(sz * scale), int(sz * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h >= sz and new_w >= sz:
                top = random.randint(0, new_h - sz)
                left = random.randint(0, new_w - sz)
            else:
                top, left = 0, 0
            image = image[top:top + sz, left:left + sz]
            mask = mask[top:top + sz, left:left + sz]

        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        return image, mask
