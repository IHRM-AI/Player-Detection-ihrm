from ultralytics import YOLO

# Load custom YOLOv8 model
model = YOLO("weights/best.pt")

# Tracking configuration
tracker = "botsort.yaml"
conf_threshold = 0.25
iou_threshold = 0.5

# Track on broadcast camera video
results1 = model.track(
    source="videos/broadcast.mp4",
    tracker=tracker,
    conf=conf_threshold,
    iou=iou_threshold,
    save=True,
    project="runs",
    name="broadcast"
)
for res in results1:
    pass

# Track on tactical camera video
results2 = model.track(
    source="videos/tacticam.mp4",
    tracker=tracker,
    conf=conf_threshold,
    iou=iou_threshold,
    save=True,
    project="runs",
    name="tacticam"
)
for res in results2:
    pass
