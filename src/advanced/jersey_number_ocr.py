import easyocr
import os
import cv2
from pathlib import Path

reader = easyocr.Reader(['en'])

def ocr_on_crop(crop_path):
    img = cv2.imread(str(crop_path))
    results = reader.readtext(img)
    for (bbox, text, conf) in results:
        print(f"🟢 Detected: {text} (conf: {conf:.2f}) from {crop_path.name}")
        # Optional filtering
        if conf > 0.5 and text.strip().isdigit():
            return text.strip()
    return None

# Run on a few broadcast crops
root_folder = Path("reid_embeddings/broadcast/crops")
for id_folder in root_folder.iterdir():
    if id_folder.is_dir():
        for crop in list(id_folder.glob("*.jpg"))[:3]:  # try 3 samples per player
            number = ocr_on_crop(crop)
