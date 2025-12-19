# import torch
# from torchvision import transforms
# from PIL import Image
# from timm import create_model

# CLASS_NAMES = [
#     'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
#     'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy',
#     'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Mosaic Virus',
#     'Tomato Septoria Leaf Spot', 'Tomato Spider Mites Two Spotted Spider Mite',
#     'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus'
# ]

# MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/efficientnet_b4_leaf.pth"
# IMG_SIZE = 380
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model
# model = create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASS_NAMES))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model = model.to(DEVICE)
# model.eval()

# # Transform
# transform = transforms.Compose([
#     transforms.Resize(IMG_SIZE + 32),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# def predict(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img_t = transform(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         outputs = model(img_t)
#         pred_idx = outputs.argmax(dim=1).item()
#     return CLASS_NAMES[pred_idx]

# # Test
# if __name__ == "__main__":
#     img_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Potato Early Blight/plant_64483.jpg"
#     print("Prediction:", predict(img_path))

# ##########################################################
import torch
from torchvision import transforms
from PIL import Image
from timm import create_model

CLASS_NAMES = [
    'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy',
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Mosaic Virus',
    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites Two Spotted Spider Mite',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus'
]

MODEL_PATH = "/home/aic_u3/aic_u3/ComputerVision/EfficientNet/efficientnet_b4_leaf.pth"
IMG_SIZE = 380
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image_path):
    """Returns (plant_name, disease_name) tuple"""
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        pred_idx = outputs.argmax(dim=1).item()
    result = CLASS_NAMES[pred_idx]
    # Split plant name and disease name
    plant_name, disease_name = result.split(" ", 1)
    return plant_name, disease_name

# Test
if __name__ == "__main__":
    img_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Potato Early Blight/plant_64483.jpg"
    plant, disease = predict(img_path)
    print(f"Plant: {plant} | Disease: {disease}")

