import json
import cv2
import numpy as np
import random
import os
from easydict import EasyDict as edict
import yaml
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
'''
    # test code: 
    from tutorial_dataset import MyDataset

    dataset = MyDataset()
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)  
    print(jpg.shape)
    print(hint.shape)
    
    # output: 
    50000
    burly wood circle with orange background
    (512, 512, 3)
    (512, 512, 3)
'''

# Fill this with your own path.
EDGE_ROOT_1 = 'xxx'
EDGE_ROOT_2 = 'xxx'
SEG_ROOT_1 = 'xxx'
SEG_ROOT_2 = 'xxx'

class MyDataset(Dataset):
    def __init__(self, img_dir, text_file_path, luminance_gray: str, img_h=512, device="cuda:0", use_seg=False, cia_results_dir=None):
        self.use_seg = use_seg
        if use_seg:
            self.coco_label_clip_emb = torch.load('./coco_label_hf_words_clip_512.pth').to(device).type(torch.float16)
            self.coco_label_clip_emb.detach().requires_grad_(False)
        # COCO-extend dataset with rich captions.
        self.img_dir_big = img_dir[0]
        self.img_list_big = os.listdir(img_dir[0])
        self.img_list_big_len = len(self.img_list_big)
        self.text_file_path_rich = text_file_path[0]
        with open(self.text_file_path_rich, "r") as f:
            self.text_cond_dict_rich = edict(yaml.safe_load(f))  # dict({'img_path_name':'text_caption', ...})
        print('text_cond.json[1] has been loaded.')
        if use_seg:
            self.seg_dir_big = SEG_ROOT_1
        self.device = device

        # COCO-extend dataset with brief captions.
        self.img_dir_mid = img_dir[1]
        self.text_file_path_brief = text_file_path[1]
        with open(self.text_file_path_brief, "r") as f:
            self.text_cond_dict_brief = json.load(f)  # dict({'img_path_name':'text_caption', ...})
        print('text_cond.json[2] has been loaded.')

        # Too careless and rough of L-CoDe's author to filter the black and white photos of the dataset.
        # There are at least 2598/59371 of the dataset are black and white!.
        # To filter them out:.
        all_img_name = list(self.text_cond_dict_brief.keys())
        for name in all_img_name:
            texts = self.text_cond_dict_brief[name]
            for text in texts:
                if 'black and white photo' in text or 'black and white picture' in text:
                    del self.text_cond_dict_brief[name]
                    break
                
        self.img_list_mid = list(self.text_cond_dict_brief.keys())
        self.img_list_mid_len = len(self.img_list_mid)
        if use_seg:
            self.seg_dir_mid = SEG_ROOT_2


        self.luminance_gray = luminance_gray
        self.img_h = img_h

        self.use_cia = False
        if cia_results_dir is not None:
            self.L_CoDer_cia_dir = cia_results_dir[0]
            self.L_CoDe_cia_dir = cia_results_dir[1]
            self.use_cia = True


    def __len__(self):
        return self.img_list_big_len + self.img_list_mid_len 

    def __getitem__(self, idx):
        '''
            output:
                jpg: color image, (512,512,3), RGB, [-1, 1].
                txt: text prompt, str.
                hint: grayscale image (2 channels) + edge image (1 channel). 
                      Although ugly, it contains more info with less memory cost.
                seg_clip_emb: standard semantic feature (SSF).
                cia_img: coarse image produced by Stage I VAE Encoder.
                
        '''
        seg_clip_emb = None
        if idx < self.img_list_big_len:
            color_img = cv2.imread(os.path.join(self.img_dir_big, self.img_list_big[idx]))
            prompt = self.text_cond_dict_rich[self.img_list_big[idx]]
            sam_mask_path = os.path.join(EDGE_ROOT_1, self.img_list_big[idx].replace('.', '_edge.'))
            sam_mask = cv2.imread(sam_mask_path)
            if self.luminance_gray == 'gray':
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
            elif self.luminance_gray == 'luminance':
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)[..., 0][..., np.newaxis]
            else:
                raise NotImplementedError
            gray_img = np.repeat(gray_img, 3, axis=2)

            if self.use_seg:
                catagory_mask = np.load(os.path.join(self.seg_dir_big, self.img_list_big[idx].replace('jpg', 'npy')))
                catagory_mask = torch.tensor(catagory_mask, dtype=torch.int32)  # use torch.tensor to save time.
                seg_clip_emb = self.coco_label_clip_emb[catagory_mask].detach().requires_grad_(False) 
            gray_img[:,:,2] = sam_mask[..., 0]

            # Normalize source images to [0, 1].
            gray_img = gray_img.astype(np.float32) / 255.0
            

            # Normalize target images to [-1, 1].
            color_img = (color_img.astype(np.float32) / 127.5) - 1.0  
            color_img = cv2.resize(color_img, (self.img_h, self.img_h))
            gray_img = cv2.resize(gray_img, (self.img_h, self.img_h))

            if self.use_cia:
                cia_img = cv2.imread(os.path.join(self.L_CoDer_cia_dir, self.img_list_big[idx]))[..., ::-1] / 255.0
        else:
            color_img = cv2.imread(os.path.join(self.img_dir_mid, self.img_list_mid[idx-self.img_list_big_len]))
            prompts = self.text_cond_dict_brief[self.img_list_mid[idx-self.img_list_big_len]]
            prompt_idx = np.random.randint(0, len(prompts))
            prompt = prompts[prompt_idx]
            sam_mask_path = os.path.join(EDGE_ROOT_2, self.img_list_mid[idx-self.img_list_big_len].replace('.', '_edge.'))
            sam_mask = cv2.imread(sam_mask_path)

            if self.luminance_gray == 'gray':
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                gray_img_one = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
            elif self.luminance_gray == 'luminance':
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                gray_img_one = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)[..., 0][..., np.newaxis]
            else:
                raise NotImplementedError
            gray_img = np.repeat(gray_img_one, 3, axis=2)


            if self.use_seg:
                catagory_mask = np.load(os.path.join(self.seg_dir_mid, self.img_list_mid[idx-self.img_list_big_len].replace('jpg', 'npy')))
                catagory_mask = torch.tensor(catagory_mask, dtype=torch.int32)  # use torch.tensor to save time.
                seg_clip_emb = self.coco_label_clip_emb[catagory_mask].detach().requires_grad_(False)

            gray_img[:,:,2] = sam_mask[..., 0]

            # Normalize source images to [0, 1].
            gray_img = gray_img.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            color_img = (color_img.astype(np.float32) / 127.5) - 1.0  
            color_img = cv2.resize(color_img, (self.img_h, self.img_h))
            gray_img = cv2.resize(gray_img, (self.img_h, self.img_h))


            cia_img = None
            if self.use_cia:
                cia_img = cv2.imread(os.path.join(self.L_CoDe_cia_dir, self.img_list_mid[idx-self.img_list_big_len].replace('.jpg', '_{}.png'.format(prompt_idx))))[..., ::-1] / 255.
                cia_img = cv2.resize(cia_img, (self.img_h, self.img_h))

    
        return dict(jpg=color_img, txt=prompt, hint=gray_img, seg_clip_emb=seg_clip_emb, cia_img=cia_img)

