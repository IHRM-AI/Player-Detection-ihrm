import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def extract_crops(video_path, yolo_model_path, output_dir, run_name):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(yolo_model_path)

    results = model.track(source=video_path, tracker="botsort.yaml", save=False, stream=True)

    frame_id = 0
    for result in tqdm(results):
        frame = result.orig_img

        # Skip if no boxes or tracking IDs available
        if not hasattr(result, "boxes") or result.boxes is None:
            continue
        if not hasattr(result.boxes, "id") or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for bbox, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            crop_dir = os.path.join(output_dir, f"id_{obj_id}")
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"{run_name}_frame{frame_id}.jpg")
            cv2.imwrite(crop_path, crop)

        frame_id += 1

#  Example usage
if __name__ == "__main__":
    extract_crops(
        video_path="videos/broadcast.mp4",
        yolo_model_path="weights/best.pt",
        output_dir="reid_embeddings/broadcast/crops",
        run_name="broadcast"
    )

    extract_crops(
        video_path="videos/tacticam.mp4",
        yolo_model_path="weights/best.pt",
        output_dir="reid_embeddings/tacticam/crops",
        run_name="tacticam"
    )