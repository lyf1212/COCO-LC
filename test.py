import os 
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from transformers import AutoTokenizer
import json
import cv2
from transformers import CLIPTextModel

# Fill this with your own path.
WHERE_IS_YOUR_SD1_5 = 'XXX'

def validation(img_dir_path, text_path, controlnet_path, save_img_path, vae, text_encoder, tokenizer, unet, 
               cia_img_dir_path=None, edge_dir_path=None, seg_dir_path=None, weight_dtype=torch.float16,
               use_ddim=True, use_cia=True, use_edge=False, use_seg=False, 
               guidance_scale=7.5, latent=None, fantastic_neg_prompt=False):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=weight_dtype)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        WHERE_IS_YOUR_SD1_5,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=None,
        torch_dtype=weight_dtype,
    )
    if use_ddim:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    pipeline.enable_xformers_memory_efficient_attention()
    with open(text_path, "r") as f:
        text_dict = json.load(f)
    print("text.json loaded done.")

    if use_seg:
        coco_label_clip_emb = torch.load('./coco_label_hf_words_clip_512.pth').to("cuda")
        print("load std semantic features successfully.")
    
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
        
    color_words = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'white', 'brown', 'pink', 'magenta', 'silver', 'olive', 'blonde', 'black', 'gold', 'colorful']
    
    for img_name_suffix, text in list(text_dict.items()):

        if 'ImageNet' in img_dir_path:
            img_name = img_name_suffix.replace('.JPEG', '.png')
            img_path = img_name
        else:
            img_path = img_name_suffix.split('_')[0]  # 'xxx.jpg'.
            idx = img_name_suffix.split('_')[1]
            img_name = img_path.replace('.jpg', '_{}.png'.format(idx))
        
        img = cv2.imread(os.path.join(img_dir_path, img_name))
        img = cv2.resize(img, dsize=(512, 512))
        if use_cia:
            # by default to be 512x512, named "gray_img" to keep consistent in the following.
            gray_img = cv2.imread(os.path.join(cia_img_dir_path, img_path.replace('.jpg', '_{}.png'.format(idx))))[..., ::-1] / 255.
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[..., 0] / 255.
            gray_img = gray_img[..., np.newaxis]
            gray_img = np.repeat(gray_img, 3, axis=2)
        if use_edge:
            # merge a consistent version, take a 6 channels=[gray_img(3), edge_img(3)] as input.
            edge_img = cv2.imread(os.path.join(edge_dir_path, img_path.replace('.', '_edge.').replace('JPEG', 'jpg'))) / 255.
            edge_img = cv2.resize(edge_img, dsize=(512, 512))
            gray_img = np.concatenate([gray_img, edge_img], axis=-1)

        img_l = cv2.cvtColor(np.float32(img)/255, cv2.COLOR_BGR2Lab)[..., 0]
        img_l = img_l[..., np.newaxis]
        
        print(text)

        seg_clip_emb = None    
        if use_seg:
            catagory_mask = torch.tensor(np.load(os.path.join(seg_dir_path, img_path.replace('.jpg','.npy').replace('.JPEG', '.npy'))), dtype=torch.long, device="cuda")
            seg_clip_emb = coco_label_clip_emb[catagory_mask]
            seg_clip_emb = seg_clip_emb.permute(2,0,1).unsqueeze(0).type(weight_dtype)

            
        text_no_color_words = []
        for word in text.split(' '):
            if word not in color_words:
                text_no_color_words.append(word)
        text_no_color = ' '.join(text_no_color_words)
        
        negative_prompt = "a black and white photo of " + text_no_color
        if fantastic_neg_prompt:
            negative_prompt = "a black and white photo, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        
        with torch.no_grad():
            img_colorized = pipeline(text, gray_img[None, ...], num_inference_steps=10, 
                                    negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                                    seg_clip_emb=seg_clip_emb, latents=latent).images[0]
        img_colorized = np.asarray(img_colorized)
        '''
            Luminance Replace Post-processing.
        '''
        img_ab = cv2.cvtColor(np.float32(img_colorized)/255, cv2.COLOR_RGB2Lab)[..., 1:]

        img_result = np.concatenate((img_l, img_ab), axis=2)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_Lab2BGR) * 255
        
        cv2.imwrite(os.path.join(args.output_dir, img_path.replace('.jpg', '_{}.png'.format(idx))), img_result)
    

if __name__ == '__main__':
    weight_dtype = torch.float16
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./test_images")
    parser.add_argument('--text_file', type=str, default="./test.json")
    parser.add_argument('--output_dir', type=str, default="results_Colorization_by_cocolc/")
    parser.add_argument('--seg_dir', type=str, default="./semantic_features")
    parser.add_argument('--edge_dir', type=str, default="./edges")
    parser.add_argument('--cia_dir', type=str, default="./test_vae_CIA_results")
    parser.add_argument('--cfg_scale', type=float, default=7.5)
    parser.add_argument('--fantastic_neg_prompt', type=bool, default=False)
    args = parser.parse_args()

    # Load models.
    text_encoder = CLIPTextModel.from_pretrained(
        WHERE_IS_YOUR_SD1_5, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(WHERE_IS_YOUR_SD1_5, subfolder="vae", revision=None, torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        WHERE_IS_YOUR_SD1_5, subfolder="unet", revision=None, torch_dtype=weight_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
            WHERE_IS_YOUR_SD1_5,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    
    # generator = torch.Generator(torch.device("cuda")).manual_seed(3183878711)
    # start_noise = torch.randn(
    #     (1, 4, 64, 64),
    #     device = "cuda",
    #     generator = generator,
    #     dtype = weight_dtype
    # )
    # torch.save(start_noise, './start_noise.pth')

    start_noise = torch.load('./start_noise.pth')

    
    with torch.no_grad():
        validation(img_dir_path=args.input_dir, 
        cia_img_dir_path=args.cia_dir,
        edge_dir_path=args.edge_dir,
        seg_dir_path=args.seg_dir,
        text_path=args.text_file,
        controlnet_path="./ckpt/controlnet",
        save_img_path=args.output_dir, 
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet, 
        use_ddim=False,
        use_cia=True,
        use_edge=True,
        use_seg=True,
        weight_dtype=weight_dtype,
        latent=start_noise,
        guidance_scale=args.cfg_scale,
        fantastic_neg_prompt=args.fantastic_neg_prompt)
