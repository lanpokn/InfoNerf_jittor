import jittor as jt
import jittor.nn as nn
import glob
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import toml
import sys

from config import *
from nerf_base import *
from utils import *
from load_blender import *
import loss as ls
##nerf comes from others code, not changed
        
class Infonerf:
    def __init__(self, cfg_path, base_path,ckptpath) -> None:
        self.cfg = set_config(cfg_path, base_path,ckptpath)
        data_cfg = self.cfg['dataset']
        train_cfg = self.cfg['training']
        data_cfg.update(train_cfg)
        self.loaded_data = load_blender_data_ex(data_cfg)
        self.img_h, self.img_w, self.focal = self.loaded_data['calib']
        print(f"Loaded blender: {data_cfg['datadir']}")
        self.exp_path = prepare_dir(self.cfg)
        print('Exp path: ', self.exp_path)
        self.model, self.model_fine, self.embed_fn, self.embeddirs_fn = create_nerf(self.cfg)
        self.get_near_c2w = GetNearC2W(self.cfg['info_loss'])
        self.loss_fn = {}
        self.init_loss()
        op_param = list(self.model.parameters()) + (list(self.model_fine.parameters()) if self.model_fine is not None else [])
        self.optimizer = jt.optim.Adam(params=op_param, lr=self.cfg['training']['lr'], betas=(0.9, 0.999))
        self.it_time = 0
        self.load_ckpt()
    #load a specific model
    def load_ckpt(self):
        if self.cfg['training']['ckpt'] != "":
            #TODO, need specific?
            model_name = self.cfg['training']['ckpt']
            self.model.load(model_name)
            print(f"Load model parameters: {model_name}")
            if self.model_fine is not None:
                model_fine_name = self.cfg['training']['ckpt']
                self.model_fine.load(model_fine_name)
                print(f"Load fine model parameters: {model_fine_name}")
    def init_loss(self):
        if self.cfg['entropy_loss']['use']:
            self.loss_fn['ent'] = ls.EntropyLoss(self.cfg['entropy_loss'])
        if self.cfg['info_loss']['use']:
            self.loss_fn['kl_smooth'] = ls.SmoothingLoss(self.cfg['info_loss'])
            self.info_lambda = self.cfg['info_loss']['info_lambda']
        self.loss_fn['img_loss'] = nn.MSELoss()
    def train(self):
        N_sample = self.cfg['rendering']['N_samples']
        N_refine = self.cfg['rendering']['N_importance']
        N_rays = self.cfg['training']['N_rand']
        N_entropy = self.cfg['entropy_loss']['N_entropy']
        self.model.train()
        if self.model_fine is not None:
            self.model_fine.train()
        for it in range(self.cfg['training']['N_iters']):
            sys.stdout.write(f"\riteration: {it}")
            sys.stdout.flush()
            self.it_time = it
            train_data = self.gen_train_data()
            all_rays_o = []
            all_rays_d = []
            for _entry in ['rays_o', 'rays_o_near', 'rays_o_ent', 'rays_o_ent_near']:
                if _entry in train_data:
                    all_rays_o.append(train_data[_entry])
            for _entry in ['rays_d', 'rays_d_near', 'rays_d_ent', 'rays_d_ent_near']:
                if _entry in train_data:
                    all_rays_d.append(train_data[_entry])
            all_rays_o = jt.concat(all_rays_o, dim=0)  # [N, 3]
            all_rays_d = jt.concat(all_rays_d, dim=0)  # [N, 3]

            render_out = self.render_rays(all_rays_o, all_rays_d, N_sample, N_refine)

            total_loss = 0.
            loss_dict = {}
            # image loss
            gt_rgb = train_data['target_rgb']
            predict_rgb = render_out['coarse']['rgb_map'][:N_rays]
            img_loss = self.loss_fn['img_loss'](gt_rgb, predict_rgb)
            loss_dict['img_loss'] = img_loss.item()
            total_loss += img_loss
            if 'fine' in render_out:
                predict_rgb_fine = render_out['fine']['rgb_map'][:N_rays]
                img_loss_fine = self.loss_fn['img_loss'](gt_rgb, predict_rgb_fine)
                loss_dict['img_loss_fine'] = img_loss_fine.item()
                total_loss += img_loss_fine

            # Ray Entropy Minimiation Loss
            ent_iter = (it < self.cfg['entropy_loss']['entropy_end_iter']) if self.cfg['entropy_loss']['entropy_end_iter'] > 0 else True
            if self.cfg['entropy_loss']['use'] and ent_iter:
                alpha_raw = render_out['fine']['alpha'] \
                    if 'fine' in render_out else render_out['coarse']['alpha']
                acc_raw = render_out['fine']['acc_map'] \
                    if 'fine' in render_out else render_out['coarse']['acc_map']

                need_remove_nearby = self.cfg['info_loss']['use'] and self.cfg['entropy_loss']['entropy_ignore_smoothing']
                near_remove_normal = (not self.cfg['entropy_loss']['computing_entropy_all']) or self.cfg['entropy_loss']['N_entropy'] <= 0
                # only compute loss for rays + unseen rays, no nearby rays
                if need_remove_nearby:
                    alpha_raw = jt.concat([alpha_raw[:N_rays], alpha_raw[2*N_rays:2*N_rays+N_entropy]], dim=0)
                    acc_raw = jt.concat([acc_raw[:N_rays], acc_raw[2*N_rays:2*N_rays+N_entropy]], dim=0)
                    if near_remove_normal:
                        alpha_raw = alpha_raw[self.N_rays:]
                        sigma = sigma[self.N_rays:]
                elif self.cfg['info_loss']['use']:
                    if near_remove_normal:
                        alpha_raw = alpha_raw[2*N_rays:]
                        acc_raw = acc_raw[2*N_rays:]
                elif near_remove_normal:
                    alpha_raw = alpha_raw[N_rays:]
                    acc_raw = acc_raw[N_rays:]

                entropy_ray_zvals_loss = self.loss_fn['ent'].ray_zvals(alpha_raw, acc_raw)
                loss_dict['entropy_ray_zvals'] = entropy_ray_zvals_loss.item()
                total_loss += self.cfg['entropy_loss']['entropy_ray_zvals_lambda'] * entropy_ray_zvals_loss

            # Infomation Gain Reduction Loss
            info_iter = (it < self.cfg['info_loss']['info_end_iter']) if self.cfg['info_loss']['info_end_iter'] > 0 else True
            if self.cfg['info_loss']['use'] and info_iter:
                alpha_raw = render_out['fine']['alpha'] \
                    if 'fine' in render_out else render_out['coarse']['alpha']
                if self.cfg['entropy_loss']['use']:
                    alpha_1 = jt.concat([alpha_raw[:N_rays], alpha_raw[2*N_rays:2*N_rays+N_entropy]], dim=0)
                    alpha_2 = jt.concat([alpha_raw[N_rays:2*N_rays], alpha_raw[2*N_rays+N_entropy:]], dim=0)
                    info_loss = self.loss_fn['kl_smooth'](alpha_1, alpha_2)
                else:
                    info_loss = self.loss_fn['kl_smooth'](alpha_raw[:N_rays], alpha_raw[N_rays:2*N_rays])
                loss_dict['KL_loss'] = info_loss.item()
                total_loss += self.info_lambda * info_loss
            loss_dict.update({"loss": total_loss.item()})
            jt.sync_all(True)
            self.optimizer.step(total_loss)

            # update learning rate
            new_lr = lr=self.cfg['training']['lr'] * (0.1 ** (it / self.cfg['training']['lr_decay']))
            self.optimizer.lr = new_lr

            # adjust lambda of Infomation Gain Reduction Loss
            if it > 0 and it % self.cfg['info_loss']['reduce_step_size'] == 0 and self.cfg['info_loss']['use']:
                self.info_lambda *= self.cfg['info_loss']['reduce_step_rate']
            
            if it % self.cfg['training']['i_print'] == 0 and it > 0:
                print(f"ITER {it}", loss_dict)
            jt.sync_all(True)
            jt.gc()

            if it % self.cfg['training']['i_testset'] == 0:
                test_save = os.path.join(self.exp_path, 'result', f"test_{it}")
                #here change to all 8 to coincide with hw
                self.run_testset(test_save, 8 if it > 0 else 50, 2)
                self.model.train()
                if self.model_fine is not None:
                    self.model_fine.train()

            # if it % self.cfg['training']['i_weights'] == 0 and it > 0:
            if it % self.cfg['training']['i_weights'] == 0:
                print("Save ckpt")
                self.model.save(os.path.join(self.exp_path, 'ckpt', f"model{it}.pkl"))
                if self.model_fine is not None:
                    self.model_fine.save(os.path.join(self.exp_path, 'ckpt', f"model_fine{it}.pkl"))