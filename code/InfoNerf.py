#rewrite here
#reference:https://github.com/itoshiko/InfoNeRF-jittor
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
import ray_utils as ru
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
        def generate_fine_model_name(model_name):
            base_name, extension = model_name.rsplit('.', 1)
            
            # Find the index of "model" in the base name
            model_index = base_name.find("model")
            
            # Check if "model" is found and insert "_fine" after it
            if model_index != -1:
                fine_model_name = f"{base_name[:model_index + 5]}_fine{base_name[model_index + 5:]}.{extension}"
                return fine_model_name
            else:
                # If "model" is not found, return the original name
                return model_name
        if self.cfg['training']['ckpt'] != "":
            model_name = self.cfg['training']['ckpt']
            self.model.load(model_name)
            print(f"Load model parameters: {model_name}")
            if self.model_fine is not None:
                model_fine_name = generate_fine_model_name(model_name)
                self.model_fine.load(model_fine_name)
                print(f"Load fine model parameters: {model_fine_name}")
    def init_loss(self):
        if self.cfg['entropy_loss']['use']:
            self.loss_fn['ent'] = ls.EntropyLoss(self.cfg['entropy_loss'])
        if self.cfg['info_loss']['use']:
            self.loss_fn['kl_smooth'] = ls.SmoothingLoss(self.cfg['info_loss'])
            self.info_lambda = self.cfg['info_loss']['info_lambda']
        self.loss_fn['img_loss'] = nn.MSELoss()
    def gen_train_data(self):
        def random_sample_ray(H, W, focal, c2w, cnt, pix_coord=None, center_crop=1.0):
            """Sample rays in target view

            Args:
                H (int): image height
                W (int): image width
                focal (float): focal length
                c2w (jt.Var): transform matrix from camera to world
                cnt (int): number of sample points
                pix_coord (jt.Var, optional): pixel coords of sample points. If None, random 
                points will be sampled. Defaults to None.
                center_crop (float, optional): sample in center area. Defaults to 1.0.

            Returns:
                tuple: rays_o ([cnt, 3]), rays_d ([cnt, 3]), 
                pix_coord ([cnt, 2], in pixel coords, unnormalized)
            """
            # generate pixel coord
            if pix_coord is None:
                sample_x = jt.rand(cnt, dtype=jt.float32) * W * center_crop
                sample_y = jt.rand(cnt, dtype=jt.float32) * H * center_crop
                sample_x += ((1 - center_crop) / 2.) * W
                sample_y += ((1 - center_crop) / 2.) * H  # [N, ]
                pix_coord = jt.stack([sample_x, sample_y], dim=-1)
            else:
                sample_x = pix_coord[..., 0]
                sample_y = pix_coord[..., 1]

            dirs = jt.stack([(sample_x-W*.5)/focal, -(sample_y-H*.5)/focal, -jt.ones_like(sample_x)], -1)  # [N, 3]
            # Rotate ray directions from camera frame to the world frame
            rays_d = jt.matmul(dirs, jt.transpose(c2w[:3, :3], (1, 0)))
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].unsqueeze(0).repeat((dirs.shape[0], 1))  # [N, 3]
            return rays_o, rays_d, pix_coord
        def sample_info_gain_rays(img_h, img_w, focal, rgb_pose, N_rand, coord, sampling_method, loaded_ray):
            if sampling_method == 'near_pose':
                rgb_near_pose = loaded_ray.get_near_c2w(rgb_pose)
                rays_o_near, rays_d_near, _ = random_sample_ray(
                    img_h, img_w, focal, rgb_near_pose, N_rand, pix_coord=coord)
            else:
                rays_o_near, rays_d_near, _ = ru.sample_nearby_ray(
                    img_h, img_w, focal, rgb_pose, coord, loaded_ray['info_loss_pixel_range'])
            loaded_ray.update({'rays_o_near': rays_o_near, 'rays_d_near': rays_d_near})
        def sample_target_rgb(coord, target, loaded_ray):
            coord = ru.normalize_pts(coord, self.img_h, self.img_w)  # normalize to [-1, 1] for sampling
            coord = coord.unsqueeze(0).unsqueeze(0)  # [1, 1, N_rand, 2]
            target_rgb = nn.grid_sampler_2d(target, coord, 'bilinear', 'zeros', False)  # [1, 3, 1, N_rand]
            target_rgb = target_rgb.squeeze(0).squeeze(-2)
            target_rgb = target_rgb.permute(1, 0)  # [N_rand, 3]
            loaded_ray.update({'target_rgb': target_rgb})
        def sample_unseen_rays(img_h, img_w, focal, rgb_pose_e, n_entropy, center_crop, loaded_ray):
            img_e = np.random.choice(self.loaded_data['imgs'].shape[0])
            rgb_pose_e = self.loaded_data['poses'][img_e]  # [4, 4]
            rays_o_ent, rays_d_ent, coord_ent = random_sample_ray(
                img_h, img_w, focal, rgb_pose_e, n_entropy, center_crop=center_crop)
            loaded_ray.update({'rays_o_ent': rays_o_ent, 'rays_d_ent': rays_d_ent})
        def sample_unseen_info_gain_rays(img_h, img_w, focal, rgb_pose_e, n_entropy, coord_ent, sampling_method, loaded_ray):
            if sampling_method == 'near_pose':
                ent_near_pose = loaded_ray.get_near_c2w(rgb_pose_e)
                rays_o_ent_near, rays_d_ent_near, _ = random_sample_ray(
                    img_h, img_w, focal, ent_near_pose, n_entropy, pix_coord=coord_ent)
            else:
                rays_o_ent_near, rays_d_ent_near, _ = ru.sample_nearby_ray(
                    img_h, img_w, focal, rgb_pose_e, coord_ent, loaded_ray['info_loss_pixel_range'])
            loaded_ray.update({'rays_o_ent_near': rays_o_ent_near, 'rays_d_ent_near': rays_d_ent_near})
        
        sample_info_gain = self.cfg['info_loss']['use']
        sample_entropy = self.cfg['entropy_loss']['use']
        loaded_ray = {}

        # Random from one image
        i_train = self.loaded_data['i_split'][0]
        img_i = np.random.choice(i_train)
        target = self.loaded_data['imgs'][img_i:img_i+1]  # [1, 3, H, W]
        rgb_pose = self.loaded_data['poses'][img_i]  # [4, 4]

        cfg_train = self.cfg['training']
        _crop = cfg_train['precrop_frac'] if self.it_time < cfg_train['precrop_iters'] else 1.0
        # sample rays in target view, ([N_rand, 3], [N_rand, 3], [N_rand, 2])
        rays_o, rays_d, coord = random_sample_ray(
            self.img_h, self.img_w, self.focal, rgb_pose, 
            self.cfg['training']['N_rand'], center_crop=_crop)
        
        loaded_ray.update({'coord': coord, 'rays_o': rays_o, 'rays_d': rays_d})
        if sample_info_gain:
            sample_info_gain_rays(self.img_h, self.img_w, self.focal, loaded_ray['rgb_pose'], 
                                self.cfg['training']['N_rand'], loaded_ray['coord'], 
                                self.cfg['info_loss']['sampling_method'], loaded_ray)
        sample_target_rgb(coord, target, loaded_ray)
        if sample_entropy:
            n_entropy = self.cfg['entropy_loss']['N_entropy']
            sample_unseen_rays(self.img_h, self.img_w, self.focal, self.loaded_data['poses'], 
                                n_entropy, _crop, loaded_ray)
        if sample_info_gain:
            sample_unseen_info_gain_rays(self.img_h, self.img_w, self.focal, 
                                        loaded_ray['rgb_pose_e'], n_entropy, 
                                        loaded_ray['coord_ent'], 
                                        self.cfg['info_loss']['sampling_method'], loaded_ray)
        return loaded_ray
    def render_rays(self, rays_o, rays_d, N_samples, N_importance=0, eval=False):
        """Volumetric rendering for rays
        """
        result = {}
        # calculate sample points along rays
        near, far = self.cfg['rendering']['near'], self.cfg['rendering']['far']
        pts, z_vals = ru.generate_pts(
            rays_o, rays_d, near, far, N_samples, 
            perturb=((not eval) and (self.cfg['rendering']['perturb'])))
        if self.cfg['rendering']['use_viewdirs']:
            view_dirs = rays_d / jt.norm(rays_d, dim=-1, keepdims=True)
            view_dirs = view_dirs.unsqueeze(1)
            view_dirs = jt.repeat(view_dirs, (1, N_samples, 1))  # [N_rays, N_samples, 3]
        else:
            view_dirs = None
        raw_out = run_network(
            pts, view_dirs, self.model, self.embed_fn, self.embeddirs_fn, 
            self.cfg['training']['netchunk'] if not eval else self.cfg['training']['evalchunk'])
        decoded_out = ru.raw2outputs(
            raw_out, z_vals, rays_d, 
            self.cfg['rendering']['raw_noise_std'] if not eval else 0., 
            self.cfg['dataset']['white_bkgd'],
            out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))
        result.update({'coarse': decoded_out})

        if N_importance > 0:
            weights = decoded_out['weights']
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = ru.sample_pdf(
                z_vals_mid, weights[...,1:-1], N_importance, 
                det=((not eval) and (self.cfg['rendering']['perturb'])))
            z_samples = z_samples.stop_grad()
            _, z_vals_re = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)
            
            # [N_rays, N_samples + N_importance, 3]
            pts_re = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_re[..., :, None]
            if view_dirs is not None:
                # [N_rays, N_samples + N_importance, 3]
                view_dirs_re = jt.repeat(view_dirs[:, :1, :], (1, N_samples + N_importance, 1))
            raw_re = run_network(
                pts_re, view_dirs_re, 
                self.model_fine if self.model_fine is not None else self.model, 
                self.embed_fn, self.embeddirs_fn, 
                self.cfg['training']['netchunk'] if not eval else self.cfg['training']['evalchunk'])
            decoded_out_re = ru.raw2outputs(
                raw_re, z_vals_re, rays_d, 
                self.cfg['rendering']['raw_noise_std'] if not eval else 0., 
                self.cfg['dataset']['white_bkgd'],
                out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))
            result.update({'fine': decoded_out_re})

        return result

    def render_image(self, poses, downsample=0, save_dir=None):
        with jt.no_grad():
            n_views = poses.shape[0]
            render = []
            if downsample != 0:
                render_h = self.img_h // (2 ** downsample)
                render_w = self.img_w // (2 ** downsample)
            else:
                render_h, render_w = self.img_h, self.img_w
            focal = self.focal * render_h / self.img_h
            for vid in tqdm(range(n_views)):
                pose = poses[vid]
                rays_o, rays_d = ru.get_rays(render_h, render_w, focal, pose)
                rays_o = rays_o.reshape((render_h * render_w, -1))
                rays_d = rays_d.reshape((render_h * render_w, -1))

                total_rays = render_h * render_w
                group_size = self.cfg['training']['chunk']
                group_num = (
                    (total_rays // group_size) if (total_rays % group_size == 0) else (total_rays // group_size + 1))
                if group_num == 0:
                    group_num = 1
                    group_size = total_rays

                group_output = []
                for gi in range(group_num):
                    start = gi * group_size
                    end = (gi + 1) * group_size
                    end = (end if (end <= total_rays) else total_rays)
                    render_out = self.render_rays(
                        rays_o[start:end], rays_d[start:end], self.cfg['rendering']['N_samples'], 
                        self.cfg['rendering']['N_importance'], True)
                    render_result = render_out['fine'] if 'fine' in render_out else render_out['coarse']
                    group_output.append(render_result['rgb_map'])
                image_rgb = jt.concat(group_output, 0)
                image_rgb = image_rgb.reshape((render_h, render_w, 3))
                render.append(image_rgb.permute(2, 0, 1))  # [3, h, w]
                if save_dir is not None:
                    predict = image_rgb.detach().numpy()
                    _img = np.clip(predict, 0., 1.)
                    _img = (_img * 255).astype(np.uint8)
                    cv.imwrite(f"{save_dir}/test_{vid}.png", _img)
            return render       
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
                test_save = os.path.join(self.exp_path, 'render_result', f"test_{it}")
                #here change to all 8 to coincide with hw
                # self.run_testset(test_save, 8 if it > 0 else 50, 2)
                self.run_testset(test_save, 8, 2)
                self.model.train()
                if self.model_fine is not None:
                    self.model_fine.train()

            # if it % self.cfg['training']['i_weights'] == 0 and it > 0:
            if it % self.cfg['training']['i_weights'] == 0:
                print("Save ckpt")
                self.model.save(os.path.join(self.exp_path, 'ckpt', f"model{it}.pkl"))
                if self.model_fine is not None:
                    self.model_fine.save(os.path.join(self.exp_path, 'ckpt', f"model_fine{it}.pkl"))

    def run_testset(self, save_path = None, skip = 8, downsample=2):
        self.model.eval()
        if self.model_fine is not None:
            self.model_fine.eval()
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        test_id = self.loaded_data['i_split'][2][::skip]
        test_pose = self.loaded_data['poses'][test_id]
        ref_images = self.loaded_data['imgs'][test_id]
        #this should not be an individual function
        # metric = self.test(test_pose, ref_images, not no_metric, not no_metric, False, save_path, downsample)
        if test_pose.ndim == 2:
            test_pose = test_pose.unsqueeze(0)
        predict = self.render_image(test_pose, downsample, save_dir=save_path)
        ref = nn.resize(ref_images, size=(self.img_h // (2 ** downsample), self.img_w // (2 ** downsample)), mode='bilinear')
        with jt.no_grad():
            metric = {}
            predict = jt.stack(predict, dim=0)  # [B, 3, H, W]
            if ref is not None:
                psnr_avg = ls.img2psnr_redefine(predict, ref)
                metric['psnr'] = psnr_avg.item()
        print("Test: ", metric)

    # #TODO, only 1 test function
    # def test(self, poses, ref=None, test_psnr=False, test_ssim=False, test_lpips=False, save_dir=None, downsample=2):
    #     if poses.ndim == 2:
    #         poses = poses.unsqueeze(0)
    #     predict = self.render_image(poses, downsample, save_dir=save_dir)
    #     ref = nn.resize(ref, size=(self.img_h // (2 ** downsample), self.img_w // (2 ** downsample)), mode='bilinear')
    #     with jt.no_grad():
    #         metric = {}
    #         predict = jt.stack(predict, dim=0)  # [B, 3, H, W]
    #         if ref is not None:
    #             psnr_avg = ls.img2psnr_redefine(predict, ref)
    #             metric['psnr'] = psnr_avg.item()
    #         return metric

    # def run_testset(self, save_path, skip, downsample=2, no_metric=False):
    #     self.model.eval()
    #     if self.model_fine is not None:
    #         self.model_fine.eval()
    #     os.makedirs(save_path, exist_ok=True)
    #     test_id = self.loaded_data['i_split'][2][::skip]
    #     test_pose = self.loaded_data['poses'][test_id]
    #     ref_images = self.loaded_data['imgs'][test_id]
    #     #this should not be an individual function
    #     metric = self.test(test_pose, ref_images, not no_metric, not no_metric, False, save_path, downsample)
    #     print("Test: ", metric)
