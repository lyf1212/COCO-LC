import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torch.nn as nn
import diffusers
from diffusers import AutoencoderKL
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.8, help='[0,1], larger means more colorful, alpha=1 means no merge.')
parser.add_argument('--img_root', default='./results_Colorization_by_cocolc_fan', help='path of images to be merged with gray info.')

YOUR_SD1_5 = '/mnt/netdisk/liyf/diffusers/examples/controlnet/sd1.5/'
# 'xxx'  # fill this with your own path of Stable Diffusion v1.5.

args = parser.parse_args()
if __name__ == "__main__":

    vae_std = AutoencoderKL.from_pretrained(os.path.join(YOUR_SD1_5, 'vae'), torch_dtype=torch.float32).to("cuda")
    vae_finetuned_merge = AutoencoderKL.from_pretrained('./ckpt/COCO_Decoder', torch_dtype=torch.float32).to("cuda")
    alpha = args.alpha

    img_root = args.img_root
    gt_root = './test_images'
    save_root = args.img_root + "_vae_merge_{}".format(alpha)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for img_name in tqdm(os.listdir(img_root)):
        gt_img = cv2.imread(os.path.join(gt_root, img_name))[..., ::-1]
        gt_img = cv2.resize(gt_img, dsize=(512, 512))
        gray_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2LAB)[..., 0][..., np.newaxis]
        img_l = cv2.cvtColor(np.float32(gt_img)/255, cv2.COLOR_RGB2LAB)[..., 0][..., np.newaxis]
        gray_img = np.repeat(gray_img, 3, axis=2)
        color_img = cv2.imread(os.path.join(img_root, img_name))
        color_img = cv2.resize(color_img, dsize=(512, 512))

        gray_img = torch.tensor(gray_img.transpose(2,0,1), device="cuda").unsqueeze(0).float()
        color_img = torch.tensor(color_img.transpose(2,0,1), device="cuda").unsqueeze(0).float()

        with torch.no_grad():
            gray_z0 = vae_std.encode(gray_img / 255).latent_dist.sample()
            _, gray_decoder_mids = vae_std.decode(gray_z0, return_dict=False)
            color_z0 = vae_std.encode(color_img / 255).latent_dist.sample()

            res_merge, _ = vae_finetuned_merge.decode(color_z0, gray_mids=gray_decoder_mids, alpha=1-alpha, return_dict=False)
            res_merge = res_merge[0].detach().cpu().numpy().transpose(1,2,0)
            img_colorized = res_merge
            img_ab = cv2.cvtColor(np.float32(img_colorized), cv2.COLOR_BGR2Lab)[..., 1:]

            img_result = np.concatenate((img_l, img_ab), axis=2)
            img_result = cv2.cvtColor(img_result, cv2.COLOR_Lab2BGR) * 255
            cv2.imwrite(os.path.join(save_root, img_name), img_result)        
    