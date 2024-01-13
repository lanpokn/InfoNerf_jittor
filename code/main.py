
import jittor as jt
import jittor.nn as nn
import glob
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv

from InfoNerf import *
##this should be rewrite TOTALLY
#TODO
def render_video(base_path,spc_path,ckpt_path):
    network = Infonerf(base_path,spc_path,ckpt_path)
    video_path = os.path.join(network.exp_path, 'video')
    network.run_testset(video_path, 1, 0, no_metric=True)

    length = 600
    fps = 24
    size = (800, 800)
    video = cv.VideoWriter(
        os.path.join(video_path, "render.mp4"), 
        cv.VideoWriter_fourcc('h', '2', '6', '4'), fps, size)
    for i in range(600):
        i_temp = i_temp%400
        img_index = i_temp if i_temp < 200 else 400 - i_temp
        img = cv.imread(os.path.join(video_path, f"test_{img_index}.png"))
        video.write(img)
    video.release()
    print(f"Video saved with length",length)
def check_PSNR(base_path,spc_path,ckpt_path):
    network = Infonerf(base_path,spc_path,ckpt_path)
    network.run_testset()


if __name__=='__main__':
    # jt.flags.use_cuda = 1
    # jt.flags.use_tensorcore = 1
    # jt.set_global_seed(0)
    # Network = Infonerf('configs/lego.toml','configs/base.toml',"result/lego/nn_model/model10000.pkl")
    # Network.train()
    # trainer.run_testset(os.path.join(trainer.exp_path, 'test'), 8, 0)result/lego/nn_model/model100000.pkl
    check_PSNR('configs/lego.toml','configs/base.toml',"result/lego/nn_model/model100000.pkl")
    #render_video('configs/lego.toml','configs/base.toml',"result/lego/nn_model/model100000.pkl")