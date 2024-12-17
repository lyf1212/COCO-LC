import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
import os
dataset_root = 'XXX'
result_root = 'XXX'
device = "cuda:0"
def create_sam_mask(mask_generator, img_path: str, result_name: str):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(512, 512))
    # Cuda OOM...
    # if img.shape[0] > 2048 or img.shape[1] > 2048:
    #     return
    masks = mask_generator.generate(img)
    margin = (255-100) / len(masks)
    sam_mask = np.zeros((img.shape[0], img.shape[1], 1))
    for i, dict_rec in enumerate(masks):
        sam_mask[np.where(dict_rec['segmentation'])] = (i+1)*margin + 100
    cv2.imwrite(result_name, sam_mask)
    edge_img = cv2.Canny(sam_mask, threshold1=50, threshold2=200)
    cv2.imwrite(result_name.replace('mask', 'edge'), edge_img) 


sam = sam_model_registry["vit_h"](checkpoint="segment-anything/sam_vit_h_4b8939.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

for img_name in tqdm(os.listdir(dataset_root)):
    img_path = os.path.join(dataset_root, img_name)
    img_mask_name = os.path.join(result_root, img_name.replace('.png', '_mask.png'))
    create_sam_mask(mask_generator, img_path, img_mask_name)
