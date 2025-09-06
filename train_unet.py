import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

# ===== U-Net 모델 정의 =====
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.center = CBR(512, 1024)
        self.dec4 = CBR(1024+512, 512)
        self.dec3 = CBR(512+256, 256)
        self.dec2 = CBR(256+128, 128)
        self.dec1 = CBR(128+64, 64)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(center), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)

# ===== Custom Dataset =====
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = mask.resize((256, 256), resample=Image.NEAREST) 

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# ===== 학습 세팅 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = SegmentationDataset("unet_dataset/train/images", "unet_dataset/train/masks", transform)
valid_dataset = SegmentationDataset("unet_dataset/valid/images", "unet_dataset/valid/masks", transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

model = UNet(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== 학습 루프 =====
if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/unet_experiment")

    def calculate_pixel_accuracy(pred, target):
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total

    def calculate_iou(pred, target, num_classes):
        iou_per_class = []
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                continue
            iou = intersection / union
            iou_per_class.append(iou)
        if len(iou_per_class) == 0:
            return 0.0
        return np.mean(iou_per_class)

    for epoch in range(EPOCHS):
        # === Training ===
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # === Validation ===
        model.eval()
        val_loss = 0
        total_pix_acc = 0
        total_iou = 0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                total_pix_acc += calculate_pixel_accuracy(preds, masks)
                total_iou += calculate_iou(preds.cpu(), masks.cpu(), NUM_CLASSES)

        avg_val_loss = val_loss / len(valid_loader)
        avg_pix_acc = total_pix_acc / len(valid_loader)
        avg_iou = total_iou / len(valid_loader)

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/PixelAccuracy", avg_pix_acc, epoch)
        writer.add_scalar("Metrics/IoU", avg_iou, epoch)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_pix_acc:.4f} | IoU: {avg_iou:.4f}")

    writer.close()
    torch.save(model.state_dict(), "unet_model.pt")
    print("✅ 모델 저장 완료: unet_model.pt")
