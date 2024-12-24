# COCO-LC: Colorfulness Controllable Language-based Colorization (ACM MM-24)
This is the official PyTorch code for our paper COCO-LC: Colorfulness Controllable Language-based Colorization.
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)]()
[![homepage](https://img.shields.io/badge/homepage-GitHub-179bd3)](https://github.com/lyf1212/COCO-LC/)

## Updates
🎅 Sorry for my carelessness, I made a mistake on the sharing through Google Drive. Now I correct it, and anyone can **click** the link below to achieve the checkpoint.
For feel to try our COOC-LC!

## Setup
### 🖥️ Environment Preparation
You can choose (1) create a new conda environment:
```
conda env create -f environment.yaml
conda activate cocolc
```
 or (2) download some essential packages:
```
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu118

pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
We have tested our model on CUDA11.8. You can download correct version of `torch`, `torchvision` from [this website](https://pytorch.org/get-started/previous-versions/).

### 🔥 Checkpoint download
You can download the finetuned VAE and controlnet checkpoints from [Google Drive](https://drive.google.com/drive/folders/1GZYRTN7_1i0TzQNu_neKE0CysagkQqLz?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1zRxikfUTxrqu2mZccBccGQ)(Code: ddpm) and replace the empty files in `ckpt` folder. 

You have to prepare SD1.5 by your own. For more convenience, you can follow [this website](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

### 💾 Dataset Preparation
As for **training set**, we adopt the setting of [L-CoDe](https://github.com/changzheng123/L-CoDe) and [L-CoIns](https://github.com/changzheng123/L-CoIns). Please refer to their brilliant works for more details!

As for **testing set**, we filter some black-and-white photos out of their test set, resulting in 3520 `(img_name, text prompts)` pairs from Extended COCO-Stuff (from L-CoDe), 12714 pairs from Multi-instance Dataset (from L-CoIns). We construct 5000 pairs from ImageNetVal5k, *i.e.* the first 5k images of ImageNet Validation set. We use [BLIP](https://github.com/salesforce/BLIP) as an image captioner to get text prompts of color images, which serves as the input of language-based colorization models.

We provide three json file containing `(img_name, text prompts)` pairs in `Extended_COCO_Stuff.json`, `Multi-instances.json` and `ImageNetVal5k.json`.


### 🤗 Code Preparation
We build our COCO-LC based on [diffusers](https://github.com/huggingface/diffusers), a simple but very useful diffusion libary. You shold first download diffusers:
```
git clone https://github.com/huggingface/diffusers.git
```
Then, move our codes in `hack_codes` to the folders in `diffusers` accordingly. 
Let's do them one-by-one.

**[WARNING] cp command will overwrite the target file directly. If you have some change before, make sure you have correct backup!**
```
cp hack_codes/models/controlnet.py diffusers/src/diffusers/models/controlnets

cp hack_codes/models/vae.py diffusers/src/diffusers/models/autoencoders

cp hack_codes/models/autoencoder_kl.py diffusers/src/diffusers/models/autoencoders

cp hack_codes/pipelines/pipeline_controlnet.py diffusers/src/diffusers/pipelines/controlnet
```
Then, run `pip install -e .` in your diffusers folder. This will merge our COCO-LC model into original diffusers libary.

We are going to create a new pipeline in diffusers of COCO-LC for more convenient inference. It will come soon~

### Semantic edge and standard semantic feature Preparation
Leverage [SAM](https://github.com/facebookresearch/segment-anything) as a zero-shot edge detector: `prepare_edge.py`.

Leverage [Mask2Former](https://github.com/facebookresearch/Mask2Former) as segmentation backbone: `prepare_seg_embedding.py`.
You can firstly setup SAM and Mask2Former, then change directory in our python scripts.

## 🏃‍♀️ Test

### 🌈 Generate coarse colorized image by CIA


### 🚀 Run main model!
```
python test.py --input_dir $YOUR_INPUT  --output_dir $YOUR_OUTPUT --cfg_scale 7.5 --fantastic_neg_prompt True
```
You can adjust more details in the code.


**Have fun in the colorful world created by COCO-LC**!

### 🕹️ More results with diverse colorfulness!
Fill your sd1.5 path in `test_vae_merge.py` and then run:
```
python test_vae_merge.py --alpha 0.8
```

### Train
We have provided a code of dataset in `tutorial_dataset_merge.py`. 

To train COCO-LC from scratch, first `cp train_controlnet.py /mnt/netdisk/liyf/COCO-LC/diffusers/examples/controlnet`, to merge our training codes to diffusers library.

-------

#### 🤔 Clarification of tricks:
There are some tricks using in the code. I have to clarify them.
- Classifier-free guidance scale will obviously affect results' quality. You can try `6.5` or `7.5` for diverse colorfulness. In our trail, `5.5` is too desaturated.
- Initial random noises count a lot. It has been noticed and exploit in many researches<sup><a href="#ref1">1</a>,<a href="#ref2">2</a></sup>, . We find a good noise in our mechine, and save it into `.pth`.
- Negative prompt. We remove color words in the origin text prompt and append it with `a black and white photo of`. Compared with the classic negative prompts, it works better with less color artifacts, but also harm to fantastic colors and we offer two options through `fantastic_neg_prompt`.
1. <p name = "ref1">NPNet: https://arxiv.org/abs/2411.09502 </p>
2. <p name = "ref2">SolvingDiffODE4SR: https://realpasu.github.io/SolvingDiffODE4SR_Website/</p>

If you have some good ideas or implementations, feel free to create a PR!

#### TODO
- [x] Release code of Main model
- [x] Release code of COCO-Decoder
- [x] Release code of CIA
- [ ] Upload arXiv version paper
- [ ] Provide online demo
- [ ] Update additional extensions
-------

If you have any questions, you can submit an Issue or contact Liyifan10081212@stu.pku.edu.cn.

If you find our code useful, please consider citing our paper.

```
@misc{cocolc,
  title={{COCO-LC}: Colorfulness Controllable Language-based Colorizations}, 
  author={Yifan Li and Yuhang Bai and Shuai Yang and Jiaying Liu},
  booktitle={the 32nd ACM International Conference on Multimedia},
  year={2024},
}
```

-------
