import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from train_unet import UNet  # 학습한 모델 클래스 불러오기

# ===== 설정 =====
image_path = "middle2.jpg"
model_path = "unet_model.pt"
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 이미지 불러오기 =====
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

# ===== 모델 로딩 =====
model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ===== 추론 =====
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]

# segmentation 결과 저장
np.save("pred_mask.npy", pred)

# ===== 시각화 (색상 지정) =====
colors = {
    0: (0, 0, 0),         # background - black
    1: (255, 255, 0),     # human - cyan
    2: (0, 255, 0),       # landing zone - green
    3: (255, 0, 0),       # non_landing_zone - red
    4: (0, 0, 255),       # box - blue
}

h, w = pred.shape
colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
for cls_id, color in colors.items():
    colored_mask[pred == cls_id] = color

# ===== 원본 이미지도 resize해서 같이 출력 =====
resized_image = image.resize((256, 256))

# ===== 시각화 =====
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(resized_image)

plt.subplot(1, 2, 2)
plt.imshow(resized_image)
plt.imshow(colored_mask, alpha=0.5)  # 마스크를 반투명하게 덧씌우기
plt.show()
