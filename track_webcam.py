import cv2
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# Tracking config
tracker = "botsort.yaml"
conf_threshold = 0.25
iou_threshold = 0.5

cap = cv2.VideoCapture(0)  # Using 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, tracker=tracker, conf=conf_threshold, iou=iou_threshold, persist=True)[0]
    annotated = results.plot()
    cv2.imshow("Live Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
