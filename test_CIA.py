import torch
import os
from diffusers import AutoencoderKL
import cv2
import numpy as np
from tqdm import tqdm

YOU_SD1_5_ROOT = 'XXX'
YOUR_TEST_IMG_ROOT = 'xxx'
YOUR_TEXT_FILE = 'xxx'

vae = AutoencoderKL.from_pretrained(YOU_SD1_5_ROOT, subfolder='vae')
vae.encoder.add_adain_resnet(device='cuda', gpnum=64)
vae_encoder = torch.load('ckpt/CIA/pytorch_model.bin')
vae.encoder.load_state_dict(vae_encoder)
config = 'CIA_results'
save_root = './' + 'test_vae_' + config.split('/')[-1]
if not os.path.exists(save_root):
    os.mkdir(save_root)

device = "cuda:2"
import clip
model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = model.to(dtype=torch.float32)

vae.requires_grad_(False)
vae.to(device)

import json
with open(YOUR_TEXT_FILE, 'r') as f:
    texts = dict(json.load(f))
img_root = YOUR_TEST_IMG_ROOT

for img_name, text in tqdm(texts.items()):
    
    img_name_fix = img_name
    img = cv2.imread(os.path.join(img_root, img_name_fix))
    img_L_512 = cv2.cvtColor(cv2.resize(img, dsize=(512, 512)), cv2.COLOR_BGR2LAB)[..., 0][..., np.newaxis]
    img = cv2.resize(img, dsize=(256, 256))
    img_L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0][..., np.newaxis]
    img_L = np.repeat(img_L, 3, 2)
    img_L_th = torch.from_numpy(img_L.transpose(2,0,1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tokens = clip.tokenize(text, truncate=True).to(device)
        clip_emb = model.encode_text(tokens)   
        latents = vae.encode(img_L_th/255, clip_emb).latent_dist.sample()
        img_cia_res = vae.decode(latents, return_dict=False)[0]

    img_cia_res = np.clip(img_cia_res.cpu().numpy().squeeze(0).transpose(1,2,0), 0, 1) * 255
    img_cia_res = cv2.resize(img_cia_res, dsize=(512, 512)).astype(np.uint8)
    img_ab = cv2.cvtColor(img_cia_res, cv2.COLOR_BGR2Lab)[..., 1:]
    img_result = np.concatenate((img_L_512, img_ab), axis=2)
    img_result = cv2.cvtColor(img_result, cv2.COLOR_Lab2BGR)
    cv2.imwrite(os.path.join(save_root, img_name_fix), img_result[..., ::-1])
