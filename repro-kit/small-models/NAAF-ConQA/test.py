# -------------------------------------------------------------------------------------
# Negative-Aware Attention Framework for Image-Text Matching  implementation based on SCAN
# https://github.com/CrossmodalGroup/NAAF
# "Negative-Aware Attention Framework for Image-Text Matching"
# Kun Zhang, Zhendong Mao, Quan Wang, Yongdong Zhang
#
# Writen by Kun Zhang, 2022
# -------------------------------------------------------------------------------------

from vocab import Vocabulary
import evaluation
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "checkpoint_16.pth_80.6_60.0_507.9.tar"
DATA_PATH = "../small-models-dataset"

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['conqa', 'gpt', 'gpt_abs', 'gpt_nonabs', 'coco', 'ext_coco', 'coco5k']) 
args = parser.parse_args()
evaluation.evalrank(RUN_PATH, args.experiment, data_path=DATA_PATH, split="test")
