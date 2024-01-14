
import jittor as jt
import jittor.nn as nn
import glob
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv

from InfoNerf import *

def render_video(cfg_path,base_path,ckpt_path):
    network = Infonerf(cfg_path,base_path,ckpt_path)
    video_path = os.path.join(network.exp_path, 'video')
    # network.test(video_path, 1, 0)

    length = 600
    fps = 24
    size = (800, 800)
    video = cv.VideoWriter(os.path.join(video_path, "render.mp4"), cv.VideoWriter_fourcc('h', '2', '6', '4'), fps, size)
    for i in range(600):
        i_temp = i%400
        img_index = i_temp if i_temp < 200 else 400 - i_temp
        img = cv.imread(os.path.join(video_path, f"test_{img_index}.png"))
        video.write(img)
    video.release()
    print(f"Video saved with length",length)

def check_PSNR(cfg_path,base_path,ckpt_path):
    network = Infonerf(cfg_path,base_path,ckpt_path)
    check_path = os.path.join(network.exp_path, 'check')
    network.test(check_path, 8, 2)


if __name__=='__main__':
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    jt.set_global_seed(0)
    Network = Infonerf('configs/lego.toml','configs/base.toml',"result/lego2/ckpt/model10000.pkl")
    Network.train()
    #check_PSNR('configs/lego.toml','configs/base.toml',"result/lego2/ckpt/model10000.pkl")
    # render_video('configs/lego.toml','configs/base.toml',"result/lego2/ckpt/model10000.pkl")