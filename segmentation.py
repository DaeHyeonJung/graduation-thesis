import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as maskUtils

# ===== 클래스 ID 매핑 =====
CLASS_NAME_TO_ID = {
    "background": 0,
    "human": 1,
    "landing zone": 2,
    "non_landing_zone": 3,
    "box": 4
}

# ===== 변환 함수 정의 =====
def convert_coco_to_mask(json_path, image_folder, output_folder, CLASS_NAME_TO_ID):
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

    with open(json_path, 'r') as f:
        coco = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    image_shapes = {img['id']: (img['height'], img['width']) for img in coco['images']}

    for ann in tqdm(coco['annotations'], desc=f"Converting {json_path}"):
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in coco['categories'] if cat['id'] == category_id), None)
        class_id = CLASS_NAME_TO_ID.get(category_name, 0)

        file_name = id_to_filename[image_id]
        height, width = image_shapes[image_id]

        mask_path = os.path.join(output_folder, "masks", file_name.replace(".jpg", ".png"))
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(ann['segmentation'], list):
            contours = np.array(ann['segmentation'][0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [contours], class_id)
        elif isinstance(ann['segmentation'], dict):
            rle = ann['segmentation']
            binary_mask = maskUtils.decode(rle)
            mask[binary_mask == 1] = class_id

        Image.fromarray(mask).save(mask_path)

        src_img_path = os.path.join(image_folder, file_name)
        dst_img_path = os.path.join(output_folder, "images", file_name)
        if not os.path.exists(dst_img_path):
            img = Image.open(src_img_path)
            img.save(dst_img_path)

    print(f"✅ 완료: {json_path} → {output_folder}")

# ===== 경로 설정 후 함수 실행 =====
convert_coco_to_mask(
    "graduation-thesis-segmentation-2/train/_annotations.coco.json",
    "graduation-thesis-segmentation-2/train",
    "unet_dataset/train",
    CLASS_NAME_TO_ID
)

convert_coco_to_mask(
    "graduation-thesis-segmentation-2/valid/_annotations.coco.json",
    "graduation-thesis-segmentation-2/valid",
    "unet_dataset/valid",
    CLASS_NAME_TO_ID
)

convert_coco_to_mask(
    "graduation-thesis-segmentation-2/test/_annotations.coco.json",
    "graduation-thesis-segmentation-2/test",
    "unet_dataset/test",
    CLASS_NAME_TO_ID
)
