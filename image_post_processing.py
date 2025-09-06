import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

image_path = "middle2.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

resized_image = image.resize((256, 256))

# ====== 함수 정의 ======

def distance_from_obstacles(mask, obstacle_class_ids=[1, 3, 4]):
    # 장애물 마스크
    obstacle_mask = np.zeros_like(mask, dtype=np.uint8)
    for cls_id in obstacle_class_ids:
        obstacle_mask |= (mask == cls_id).astype(np.uint8)

    # 착륙 가능 영역 (class 2)
    landing_mask = (mask == 2).astype(np.uint8)

    # 착륙 가능 영역 안에서만 거리 계산
    dist_input = (1 - obstacle_mask) * landing_mask
    dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
    return dist_map

def compute_basic_score(mask, dist_map, drone_pos, landing_class=2,
                        weight_clearance=1.0, weight_distance=0.3):
    h, w = mask.shape
    score_map = np.full((h, w), -np.inf, dtype=np.float32)

    landing_mask = (mask == landing_class)
    for y in range(h):
        for x in range(w):
            if not landing_mask[y, x]:
                continue
            clearance = dist_map[y, x]
            dist_to_drone = np.linalg.norm(np.array([y, x]) - drone_pos)
            score = weight_clearance * clearance - weight_distance * dist_to_drone
            score_map[y, x] = score

    return score_map


# ====== 실행부 ======

if __name__ == "__main__":
    # 1. 추론 결과 불러오기
    pred = np.load("pred_mask.npy")  # shape: [H, W]

    # 2. 착륙 가능 지역 마스크 만들기 (class 2)
    landing_mask = (pred == 2)

    # 3. 드론 위치 설정 (중앙)
    drone_pos = (pred.shape[0] // 2, pred.shape[1] // 2)

    # 4. 장애물 거리 맵 계산
    dist_map = distance_from_obstacles(pred, obstacle_class_ids=[1, 4])

    # 5. 점수 계산
    score_map = compute_basic_score(pred, dist_map, drone_pos,
                                    landing_class=2,
                                    weight_clearance=1.0,
                                    weight_distance=0.3)

    # 6. 최적 착륙 지점 선택
    best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
    print(f"📍 최적 착륙 위치: (x={best_x}, y={best_y})")

    # ====== 시각화 ======
    plt.figure(figsize=(7, 1))\
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(resized_image)
    plt.axis('off')

    """# (1) 착륙 가능 지역 마스크 시각화
    plt.subplot(1, 4, 2)
    plt.title("Landing Zone Mask (class 2)")
    plt.imshow(landing_mask, cmap='gray')
    plt.axis('off')"""

    # (2) 장애물 거리 맵
    plt.subplot(1, 3, 2)
    plt.title("Distance from Obstacles")
    plt.imshow(dist_map, cmap='Blues_r')
    plt.colorbar()
    plt.axis('off')

    # (3) 점수 맵 + 최적 착륙 위치
    plt.subplot(1, 3, 3)
    plt.title("Landing Score Map")
    im = plt.imshow(score_map, cmap='hot')
    plt.scatter([best_x], [best_y], c='blue', label='Best')
    plt.legend(fontsize=11, handletextpad=0.1)
    plt.colorbar(im)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
