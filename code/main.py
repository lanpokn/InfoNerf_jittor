
import jittor as jt
import jittor.nn as nn
import glob
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv

##this should be rewrite TOTALLY
#TODO
def render_video(ckpt_path):
    train_cfg = load_config('configs/lego.toml', 'configs/base.toml')
    train_cfg["training"]["ckpt"] = ckpt_path
    trainer = Trainer(train_cfg)
    video_path = os.path.join(trainer.exp_path, 'video')
    trainer.run_testset(video_path, 1, 0, no_metric=True)
    fps = 24
    size = (800, 800)
    video = cv.VideoWriter(
        os.path.join(video_path, "render.mp4"), 
        cv.VideoWriter_fourcc('h', '2', '6', '4'), fps, size)
    
    for i in range(200):
        img = cv.imread(os.path.join(video_path, f"test_{i}.png"))
        video.write(img)
    for i in range(198, -1, -1):
        img = cv.imread(os.path.join(video_path, f"test_{i}.png"))
        video.write(img)
    video.release()
    print(f"Video saved.")



if __name__=='__main__':
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    jt.set_global_seed(0)
    # train_cfg = load_config('configs/lego.toml', 'configs/base.toml')
    # train_cfg["training"]["ckpt"] = "logs/lego_ent/ckpt/"
    # trainer = Trainer(train_cfg)
    # trainer.train()
    # trainer.run_testset(os.path.join(trainer.exp_path, 'test'), 8, 0)
    # render_video("logs/lego_ent/ckpt")