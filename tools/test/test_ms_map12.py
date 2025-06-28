from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('/ultralytics//runs/obb/       /weights/best.pt')
metrics = model.val(
    data='/ultralytics/cfg/datasets/HRSC2016_MS.yaml',
    split='test',
    task="obb",
    conf=0.01,
    nms=True,
)
