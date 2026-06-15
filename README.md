# HA-LoRA

Parameter-efficient adaptation of frozen DINOv3 for remote-sensing semantic segmentation.

This repository contains the core implementation for SSD-LoRA and TCAM-enhanced
structural adaptation on LoveDA and ISPRS Potsdam. Model weights, datasets, experiment
outputs, logs, and private research notes are intentionally not tracked.

## Setup

```bash
conda create -n halora python=3.10
conda activate halora
pip install -r requirements.txt
```

Place the DINOv3 source tree at `./dinov3` or install DINOv3 so that its modules are
importable. Put pretrained checkpoints under `./checkpoints`, for example:

```text
checkpoints/
└── modelscope_dinov3_vitl16_lvd1689m/
    └── model.safetensors
```

## Data Layout

LoveDA:

```text
datasets/LoveDA/
├── Train/
│   ├── Urban/images_png, Urban/masks_png
│   └── Rural/images_png, Rural/masks_png
└── Val/
    ├── Urban/images_png, Urban/masks_png
    └── Rural/images_png, Rural/masks_png
```

ISPRS Potsdam:

```text
datasets/ISPRS_Potsdam/
├── 2_Ortho_RGB/
│   └── top_potsdam_*_RGB.tif
└── top_potsdam_*_label.tif
```

## Training

Copy an example config and edit local paths:

```bash
cp configs/potsdam_b2_tcam.example.yaml configs/potsdam_b2_tcam.yaml
python cross_dataset/train_cross.py --config configs/potsdam_b2_tcam.yaml
```

Useful examples:

```text
configs/potsdam_b2.example.yaml
configs/potsdam_b2_tcam.example.yaml
configs/loveda_b2_tcam.example.yaml
```

## Evaluation

Potsdam:

```bash
python cross_dataset/eval_potsdam.py \
  --config configs/potsdam_b2_tcam.yaml \
  --checkpoint outputs/potsdam_b2_tcam/latest_model.pth \
  --data_root ./datasets/ISPRS_Potsdam \
  --tile_size 512 \
  --stride 256 \
  --tta hflip+vflip \
  --scales 0.75,1.0,1.25,1.5
```

LoveDA:

```bash
python cross_dataset/eval_loveda.py \
  --config configs/loveda_b2_tcam.yaml \
  --checkpoint outputs/loveda_b2_tcam/latest_model.pth \
  --data_root ./datasets/LoveDA \
  --tile_size 512 \
  --stride 256 \
  --tta hflip+vflip \
  --scales 0.75,1.0,1.25,1.5
```

Use `--tta none --scales none` for a no-test-time-augmentation protocol.

## Repository Contents

```text
src/
  model.py              SSD-LoRA, TCAM, decoders, segmentation model
  dataset.py            LoveDA dataset
  dataset_potsdam.py    Potsdam dataset
cross_dataset/
  train_cross.py        Unified LoveDA/Potsdam training entry point
  eval_loveda.py        LoveDA sliding-window evaluation
  eval_potsdam.py       Potsdam sliding-window evaluation
configs/
  *.example.yaml        Portable example configs
scripts/
  train.sh, eval.sh     Simple local launch helpers
```

## Not Tracked

The `.gitignore` excludes:

```text
checkpoints/       pretrained weights and trained checkpoints
datasets/, data/   datasets
outputs/, logs/    training outputs and tensorboard logs
experiments/       experiment artifacts
refine-logs/       private research records
local_papers/      PDFs and paper notes
```
