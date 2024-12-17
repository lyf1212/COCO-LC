# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import transforms as T
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    root_dir = 'XXX'
    img_list = os.listdir(root_dir)
    for path in tqdm.tqdm(img_list):
        # use PIL, to be consistent with evaluation
        img = read_image(os.path.join(root_dir, path), format="BGR")
        input_image = T.AugInput(img)
        _ = T.Resize((512, 512))(input_image)
        img = input_image.image
        predictions, visualized_output = demo.run_on_image(img)
        # visualized_output.save('./result_{}.jpg'.format(path))
        id_mask, seg_infos = predictions['panoptic_seg']

            
        out_filename = os.path.join('XXX', os.path.basename(path))
        catagory_mask = id_mask.cpu().numpy()
        '''
            [Core of Hack codes]
        '''
        for seg_info in seg_infos:
            catagory_mask[np.where(catagory_mask==seg_info['id'])] = seg_info['category_id']
        print(catagory_mask.shape)
        np.save(out_filename.replace('.JPEG', '.npy'), catagory_mask.astype(np.uint8))
