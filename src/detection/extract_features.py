import os
import torch
import torchreid
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

torchreid.models.show_avai_models()  

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()
model.to("cpu")  # Use CPU
# Transform image to model input format
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features_from_crops(crop_folder, save_path):
    features = {}

    for id_folder in tqdm(os.listdir(crop_folder), desc="Extracting ReID Features"):
        id_path = os.path.join(crop_folder, id_folder)
        embeddings = []

        for img_file in os.listdir(id_path):
            img_path = os.path.join(id_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to("cpu")  # Use CPU

            with torch.no_grad():
                feat = model(img)
            embeddings.append(feat.cpu().numpy())

        if embeddings:
            avg_feat = np.mean(np.vstack(embeddings), axis=0)
            features[id_folder] = avg_feat

    # Save to .npy file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, features)

if __name__ == "__main__":
    extract_features_from_crops(
        crop_folder="reid_embeddings/broadcast/crops",
        save_path="reid_embeddings/broadcast/features/features.npy"
    )
    extract_features_from_crops(
        crop_folder="reid_embeddings/tacticam/crops",
        save_path="reid_embeddings/tacticam/features/features.npy"
    )
