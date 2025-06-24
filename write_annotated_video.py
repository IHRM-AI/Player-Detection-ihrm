import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("weights/best.pt")

# Tracking parameters
tracker = "botsort.yaml"
conf_threshold = 0.25
iou_threshold = 0.5

# Load the input video
cap = cv2.VideoCapture("videos/broadcast.mp4")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputs/broadcast_annotated.mp4", fourcc, fps, (width, height))

# Run tracking frame by frame
while True:
    success, frame = cap.read()
    if not success:
        break

    # Run tracking on the current frame
    results = model.track(frame, tracker=tracker, conf=conf_threshold, iou=iou_threshold, persist=True)

    # Annotate the frame
    annotated = results[0].plot()

    # Write annotated frame to video
    out.write(annotated)

# Release everything
cap.release()
out.release()
print("✅ Annotated video saved as outputs/broadcast_annotated.mp4")
