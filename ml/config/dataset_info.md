# Dataset for Image Moderation Detector
## Classes

We are detecting these objects:
- cigarette
- vape
- joint
- gun
- knife
- alcohol_bottle

## Source
- Images collected from [image-moderation-detector]
- Annotated with bounding boxes (object detection).

## Format
- Exported from Roboflow as **YOLOv8 PyTorch** dataset.
- Folder structure (after unzip):

dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data.yaml
