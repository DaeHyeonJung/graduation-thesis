# batch_landing.py
import os, glob, csv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from train_unet import UNet
from image_post_processing import distance_from_obstacles, compute_basic_score

# ───── 경로 / 하이퍼파라미터 ──────────────────────────────────
IMG_DIR   = "input_images"           # 처리 대상 폴더
VIS_DIR   = "results/vis"            # 시각화 PNG 저장
CSV_PATH  = "results/best_landing.csv"

MODEL_PATH = "unet_model.pt"
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_CLEARANCE = 1.0
WEIGHT_DISTANCE  = 0.3
OBSTACLE_IDS     = [1, 3, 4]         # 사람·랙·박스

# ───── 전처리 (모델 학습 때와 동일) ──────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

os.makedirs(VIS_DIR, exist_ok=True)

# ───── U-Net 모델 로드 ─────────────────────────────────────
model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ───── CSV 헤더 작성 ──────────────────────────────────────
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["filename", "best_x", "best_y", "score"])

# ───── 이미지 일괄 처리 루프 ──────────────────────────────
for path in sorted(glob.glob(os.path.join(IMG_DIR, "*.*"))):
    name = os.path.basename(path)

    # 1) 이미지 로드 → 텐서
    img = Image.open(path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)

    # 2) 세그멘테이션 추론
    with torch.no_grad():
        logits = model(inp)
        pred   = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()   # [H,W]

    # 3) 후처리: 거리맵·점수맵
    drone_pos = (pred.shape[0] // 2, pred.shape[1] // 2)
    dist_map  = distance_from_obstacles(pred, OBSTACLE_IDS)
    score_map = compute_basic_score(pred, dist_map, drone_pos,
                                    landing_class=2,
                                    weight_clearance=WEIGHT_CLEARANCE,
                                    weight_distance=WEIGHT_DISTANCE)

    best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
    best_score     = score_map[best_y, best_x]

    # 4) CSV 기록
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([name, best_x, best_y, f"{best_score:.3f}"])

    # 5) 시각화 PNG 저장
    plt.figure(figsize=(4,6))
    plt.imshow(score_map, cmap="hot")
    plt.scatter([best_x], [best_y], c="blue", s=20)
    plt.title(name, fontsize=8); plt.axis("off"); plt.tight_layout()
    out_png = os.path.join(VIS_DIR, os.path.splitext(name)[0] + "_score.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[✓] {name:18s}  best=({best_x:3d},{best_y:3d})  score={best_score:.2f}")

print(f"\n=== 완료! CSV → {CSV_PATH}")
