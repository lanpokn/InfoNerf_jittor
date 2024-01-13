#rewrite here
#reference:
#code in original paper
#https://github.com/itoshiko/InfoNeRF-jittor, which is translated from pytorch to jittor
#I mainly upgrade from https://github.com/itoshiko/InfoNeRF-jitto, details in readme

# In neural networks, the concept of a "fine network" is often associated with refining or enhancing the results obtained from a primary or "coarse network." Here are some common scenarios where fine networks are used:

# Two-Stage Architecture: A two-stage architecture involves using a coarse network to obtain an initial prediction and then refining that prediction using a fine network. The coarse network provides a quick estimation, and the fine network performs detailed adjustments.
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
import loss as ls
class GetNearC2W:
    def __init__(self, args):
        super(GetNearC2W, self).__init__()
        self.near_c2w_type = args['near_c2w_type']
        self.near_c2w_rot = args['near_c2w_rot']
        self.near_c2w_trans = args['near_c2w_trans']
    
    def __call__(self, c2w, all_poses=None, j=None):
        if self.near_c2w_type == 'rot_from_origin':
            return self.rot_from_origin(c2w)
        elif self.near_c2w_type == 'near':
            return self.near(c2w, all_poses)
        elif self.near_c2w_type == 'random_pos':
            return self.random_pos(c2w)
        elif self.near_c2w_type == 'random_dir':
            return self.random_dir(c2w, j)
   
    def random_pos(self, c2w):
        c2w[:3, -1] += self.near_c2w_trans * jt.randn(3)
        return c2w 
    
    def random_dir(self, c2w, j):
        rot_mat = self.get_rotation_matrix(j)
        rot = rot_mat @ c2w[:3,:3]  # [3, 3]
        c2w[:3, :3] = rot
        return c2w
    
    def rot_from_origin(self, c2w):
        rot = c2w[:3, :3]  # [3, 3]
        pos = c2w[:3, -1:]  # [3, 1]
        rot_mat = self.get_rotation_matrix()
        pos = rot_mat @ pos
        rot = rot_mat @ rot
        new_c2w = jt.zeros((4, 4), dtype=jt.float32)
        new_c2w[:3, :3] = rot
        new_c2w[:3, -1:] = pos
        new_c2w[3, 3] = 1
        return new_c2w

    def get_rotation_matrix(self):
        rotation = self.near_c2w_rot

        phi = (rotation*(np.pi / 180.))
        x = np.random.uniform(-phi, phi)
        y = np.random.uniform(-phi, phi)
        z = np.random.uniform(-phi, phi)
        
        rot_x = np.array([
            [1,0,0],
            [0,np.cos(x),-np.sin(x)],
            [0,np.sin(x), np.cos(x)]])
        rot_y = np.array([
            [np.cos(y),0,-np.sin(y)],
            [0,1,0],
            [np.sin(y),0, np.cos(y)]])
        rot_z = np.array([
            [np.cos(z),-np.sin(z),0],
            [np.sin(z),np.cos(z),0],
            [0,0,1]])
        _rot = rot_x @ (rot_y @ rot_z)
        return jt.array(_rot)
class Infonerf:
    def __init__(self, cfg_path, base_path,ckptpath) -> None:
        #load a specific model
        def load_ckpt():
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
                if self.model_fine is not None:
                    model_fine_name = generate_fine_model_name(model_name)
                    self.model_fine.load(model_fine_name)
        self.cfg = set_config(cfg_path, base_path,ckptpath)
        data_cfg = self.cfg['dataset']
        train_cfg = self.cfg['training']
        data_cfg.update(train_cfg)
        self.loaded_data = load_blender_data_ex(data_cfg)
        self.img_h, self.img_w, self.focal = self.loaded_data['calib']
        self.exp_path = prepare_dir(self.cfg)
        self.model, self.model_fine, self.embed_fn, self.embeddirs_fn = create_nerf(self.cfg)
        self.get_near_c2w = GetNearC2W(self.cfg['info_loss'])
        self.loss_fn = {}
        self.__init_loss()
        op_param = list(self.model.parameters()) + (list(self.model_fine.parameters()) if self.model_fine is not None else [])
        self.optimizer = jt.optim.Adam(params=op_param, lr=self.cfg['training']['lr'], betas=(0.9, 0.999))
        self.it_time = 0
        load_ckpt()
    
    def __init_loss(self):
        if self.cfg['entropy_loss']['use']:
            self.loss_fn['ent'] = ls.EntropyLoss(self.cfg['entropy_loss'])
        if self.cfg['info_loss']['use']:
            self.loss_fn['kl_smooth'] = ls.SmoothingLoss(self.cfg['info_loss'])
            self.info_lambda = self.cfg['info_loss']['info_lambda']
        self.loss_fn['img_loss'] = nn.MSELoss()
    def __render_image(self, poses, dsample_ratio=0, save_dir=None):
        def get_rays(H, W, focal, c2w, padding=None):
            """get rays in world coordinate of full image
            Args:
                H (int): height of target image
                W (int): width of target image
                focal (flaot): focal length of camera
                c2w (jittor.Var): transform matrix from camera coordinate to world coordinate
                padding (int, optional): padding border around the image. Defaults to None.

            Returns:
                tuple(jittor.Var[H, W, 3], jittor.Var[H, W, 3]): 
                origin of rays, direction of rays, both transformed to world coordinate
            """
            # create pts in pixel coordinate
            if padding is not None:
                i, j = jt.meshgrid(jt.linspace(-padding, W-1+padding, W+2*padding), jt.linspace(-padding, H-1+padding, H+2*padding)) 
            else:
                i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
            i = jt.transpose(i, (1, 0))
            j = jt.transpose(j, (1, 0))
            # transform to camera coordinate
            dirs = jt.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -jt.ones_like(i)], -1)  # [H, W, 3]
            # Rotate ray directions from camera frame to the world frame
            rays_d =  dirs @ jt.transpose(c2w[:3, :3], (1, 0))
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].repeat((H, W, 1))  # [H, W, 3]
            return rays_o, rays_d
        with jt.no_grad():
            n_views = poses.shape[0]
            render = []
            if dsample_ratio != 0:
                render_h = self.img_h // (2 ** dsample_ratio)
                render_w = self.img_w // (2 ** dsample_ratio)
            else:
                render_h, render_w = self.img_h, self.img_w
            focal = self.focal * render_h / self.img_h
            for vid in tqdm(range(n_views)):
                pose = poses[vid]
                rays_o, rays_d = get_rays(render_h, render_w, focal, pose)
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
                    render_out = self.__render_rays(
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
    def __gen_train_data(self):
        def sample_nearby_ray(H, W, focal, c2w, pix_coord, distance):
            new_x, new_y = pix_coord[..., 0], pix_coord[..., 1]
            pts_shape = pix_coord.shape[:-1]
            offset = np.random.randint(1, distance * 2 + 1)
            offset_x = jt.randint(0, offset + 1, pts_shape)
            offset_y = offset - offset_x
            
            new_x += offset_x
            new_y += offset_y
            new_pix_coord = jt.stack([new_x, new_y], dim=-1)

            dirs = jt.stack([(new_x-W*.5) / focal, -(new_y-H*.5) / focal, -jt.ones_like(new_x)], -1)  # [..., 3]
            # Rotate ray directions from camera frame to the world frame
            rays_d = jt.matmul(dirs, c2w[:3, :3])
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].repeat(list(dirs.shape[:-1]) + [1, ])  # [..., 3]
            return rays_o, rays_d, new_pix_coord
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
                rays_o_near, rays_d_near, _ = sample_nearby_ray(
                    img_h, img_w, focal, rgb_pose, coord, loaded_ray['info_loss_pixel_range'])
            loaded_ray.update({'rays_o_near': rays_o_near, 'rays_d_near': rays_d_near})
        def sample_target_rgb(coord, target, loaded_ray):
            def normalize_pts(pts, H, W):
                pts[..., 0] = pts[..., 0] / W
                pts[..., 1] = pts[..., 1] / H
                pts = pts * 2.0 - 1.0
                return pts
            coord = normalize_pts(coord, self.img_h, self.img_w)  # normalize to [-1, 1] for sampling
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
                rays_o_ent_near, rays_d_ent_near, _ = sample_nearby_ray(
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
    def __render_rays(self, rays_o, rays_d, N_samples, N_importance=0, eval=False):

        def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False,
                    out_alpha=False, out_sigma=False, out_dist=False, debug_save=False):
            # """Transforms model's predictions to semantically meaningful values.
            # Args:
            #     raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            #     z_vals: [num_rays, num_samples along ray]. Integration time.
            #     rays_d: [num_rays, 3]. Direction of each ray.
            # Returns:
            #     rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            #     disp_map: [num_rays]. Disparity map. Inverse of depth map.
            #     acc_map: [num_rays]. Sum of weights along each ray.
            #     weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            #     depth_map: [num_rays]. Estimated distance to object.
            # """    
            def raw2alpha(raw, dists, act_fn=nn.relu):
                return 1. - jt.exp(-act_fn(raw) * dists)
            # distance between sample points
            dists = z_vals[..., 1:] - z_vals[..., :-1] 
            dists = jt.concat([dists, jt.expand(jt.array([1e10]), dists[..., :1].shape)], -1)  # [N_rays, N_samples]
            dists = dists * jt.norm(rays_d, dim=-1, keepdims=True)  # [N_rays, N_samples]

            rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
            noise = 0.
            if raw_noise_std > 0.:
                noise = jt.randn(raw[..., 3].shape) * raw_noise_std

            # alpha: 1. - exp(-delta * sigma)
            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
            sigma = nn.relu(raw[..., 3] + noise)  # for sigma output
            # transmission, integral(sigma * delta) * alpha
            weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
            # do the integral to obtain rgb output
            rgb_map = jt.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

            depth_map = jt.sum(weights * z_vals, -1)
            disp_map = 1. / jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
            acc_map = jt.sum(weights, -1)
            
            if white_bkgd:
                rgb_map = rgb_map + (1. - acc_map[...,None])
            
            output = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map':acc_map, 'weights': weights, 'depth_map': depth_map}
            if out_alpha:
                output.update({'alpha': alpha})
            if out_sigma:
                output.update({'sigma': sigma})
            if out_dist:
                output.update({'dists': dists})
            return output
        def calculate_sample_points(rays_o, rays_d, N_samples, near, far, eval, perturb):
            """Calculate sample points along rays."""
            def generate_pts(rays_o, rays_d, near, far, N_samples, lindisp=False, perturb=False):
                near, far = near * jt.ones_like(rays_d[..., :1]), far * jt.ones_like(rays_d[..., :1])

                t_vals = jt.linspace(0., 1., steps=N_samples)
                if not lindisp:
                    z_vals = near * (1. - t_vals) + far * (t_vals)
                else:
                    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

                # not sample at fixed z, but random position betwween [z, z+1]
                if perturb:
                    # get intervals between samples
                    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                    upper = jt.concat([mids, z_vals[..., -1:]], -1)
                    lower = jt.concat([z_vals[..., :1], mids], -1)
                    # stratified samples in those intervals
                    t_rand = jt.rand(z_vals.shape)
                    z_vals = lower + (upper - lower) * t_rand
                
                pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]
                return pts, z_vals
            pts, z_vals = generate_pts(
                rays_o, rays_d, near, far, N_samples,
                perturb=((not eval) and (perturb)))
            
            return pts, z_vals
        def handle_view_directions(rays_d, N_samples, use_viewdirs):
            """Handle view directions if enabled."""
            if use_viewdirs:
                view_dirs = rays_d / jt.norm(rays_d, dim=-1, keepdims=True)
                view_dirs = view_dirs.unsqueeze(1)
                view_dirs = jt.repeat(view_dirs, (1, N_samples, 1))  # [N_rays, N_samples, 3]
            else:
                view_dirs = None

            return view_dirs
        def calculate_fine_sample_points(N_importance, weights, z_vals, eval, perturb):
            """Calculate sample points for fine rendering."""
            def sample_pdf(bins, weights, N_samples, det=False):
                # Get pdf
                weights = weights + 1e-5 # prevent nans
                pdf = weights / jt.sum(weights, -1, keepdims=True)
                cdf = jt.cumsum(pdf, -1)
                cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

                # Take uniform samples
                if det:
                    u = jt.linspace(0., 1., steps=N_samples)
                    u = u.expand(list(cdf.shape[:-1]) + [N_samples])
                else:
                    u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

                # Invert CDF
                # u = u.contiguous()
                inds = jt.searchsorted(cdf, u, right=True)
                below = jt.maximum(jt.zeros_like(inds-1), inds-1)
                above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
                inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

                # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
                # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
                matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
                cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
                bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

                denom = (cdf_g[...,1]-cdf_g[...,0])
                denom = jt.where(denom<1e-5, jt.ones_like(denom), denom)
                t = (u-cdf_g[...,0])/denom
                samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

                return samples

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance,
                det=((not eval) and (perturb)))
            z_samples = z_samples.stop_grad()
            _, z_vals_re = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)

            return z_vals_re
        result = {}

        # Calculate sample points along rays
        near, far = self.cfg['rendering']['near'], self.cfg['rendering']['far']
        pts, z_vals = calculate_sample_points(rays_o, rays_d, N_samples, near, far,
                                            eval, self.cfg['rendering']['perturb'])

        # Handle view directions if enabled
        view_dirs = handle_view_directions(rays_d, N_samples, self.cfg['rendering']['use_viewdirs'])

        # Run coarse rendering network
        raw_out = run_network(
            pts, view_dirs, self.model, self.embed_fn, self.embeddirs_fn,
            self.cfg['training']['netchunk'] if not eval else  self.cfg['training']['evalchunk'])
        # Decode and store results for coarse rendering
        decoded_out = raw2outputs(
            raw_out, z_vals, rays_d,
            self.cfg['rendering']['raw_noise_std'] if not eval else 0.,
            self.cfg['dataset']['white_bkgd'],
            out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))

        result.update({'coarse': decoded_out})

        # Run fine rendering network if N_importance > 0
        if N_importance > 0:
            weights = decoded_out['weights']
            z_vals_re = calculate_fine_sample_points(N_importance, weights, z_vals, eval, self.cfg['rendering']['perturb'])

            # Calculate sample points for fine rendering
            pts_re = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_re[..., :, None]
            if view_dirs is not None:
                view_dirs_re = jt.repeat(view_dirs[:, :1, :], (1, N_samples + N_importance, 1))
            raw_re = run_network(
                pts_re, view_dirs_re,
                self.model_fine if self.model_fine is not None else self.model,
                self.embed_fn, self.embeddirs_fn,
                self.cfg['training']['netchunk'] if not eval else self.cfg['training']['evalchunk'])
            # Decode and store results for fine rendering
            decoded_out_re = raw2outputs(
                raw_re, z_vals_re, rays_d,
                self.cfg['rendering']['raw_noise_std'] if not eval else 0.,
                self.cfg['dataset']['white_bkgd'],
                out_alpha=(not eval), out_sigma=(not eval), out_dist=(not eval))

            result.update({'fine': decoded_out_re})

        return result
       
    ## above are entrance function
    def train(self):
        # Initialize training parameters
        def collect_rays(train_data):
            # Collect ray origins and directions for rendering
            all_rays_o = concatenate_rays(train_data, ['rays_o', 'rays_o_near', 'rays_o_ent', 'rays_o_ent_near'])
            all_rays_d = concatenate_rays(train_data, ['rays_d', 'rays_d_near', 'rays_d_ent', 'rays_d_ent_near'])
            return all_rays_o, all_rays_d
        def concatenate_rays(train_data, entries):
            # Concatenate ray origins or directions based on the specified entries
            rays_list = [train_data[_entry] for _entry in entries if _entry in train_data]
            concatenated_rays = jt.concat(rays_list, dim=0)
            return concatenated_rays
        def calculate_losses(render_out, train_data, N_rays, N_entropy, it):
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
            loss_dict.update({"loss": total_loss.item()})
            jt.sync_all(True)
            return total_loss, loss_dict
        def adjust_info_lambda(it):
            # Adjust lambda of Information Gain Reduction Loss
            if it > 0 and it % self.cfg['info_loss']['reduce_step_size'] == 0 and self.cfg['info_loss']['use']:
                self.info_lambda *= self.cfg['info_loss']['reduce_step_rate']
        def print_and_save_results(it, loss_dict):
            # Print and save results periodically
            if it % self.cfg['training']['i_print'] == 0 and it > 0:
                print(f", loss==>", loss_dict["loss"])
            jt.sync_all(True)
            jt.gc()

            if it % self.cfg['training']['i_testset'] == 0:
                test_save = os.path.join(self.exp_path, 'render_result', f"test_{it}")
                self.test(test_save, 8, 2)
                self.model.train()
                if self.model_fine is not None:
                    self.model_fine.train()

            if it % self.cfg['training']['i_weights'] == 0:
                print("nn model saved as pkl")
                self.model.save(os.path.join(self.exp_path, 'ckpt', f"model{it}.pkl"))
                if self.model_fine is not None:
                    self.model_fine.save(os.path.join(self.exp_path, 'ckpt', f"model_fine{it}.pkl"))
        N_sample = self.cfg['rendering']['N_samples']
        N_refine = self.cfg['rendering']['N_importance']
        N_rays = self.cfg['training']['N_rand']
        N_entropy = self.cfg['entropy_loss']['N_entropy']
        # Set the model to training mode
        self.model.train()
        if self.model_fine is not None:
            self.model_fine.train()
        # Training loop
        for it in range(self.cfg['training']['N_iters']):
            sys.stdout.write(f"\riteration: {it}")
            sys.stdout.flush()
            self.it_time = it
            # Generate training data
            train_data = self.__gen_train_data()
            all_rays_o, all_rays_d =collect_rays(train_data)
            # Render rays and calculate losses
            render_out = self.__render_rays(all_rays_o, all_rays_d, N_sample, N_refine)
            total_loss, loss_dict =calculate_losses(render_out, train_data, N_rays, N_entropy, it)
            # Perform optimization step
            self.optimizer.step(total_loss)
            # Update learning rate
            new_lr = self.cfg['training']['lr'] * (0.1 ** (it / self.cfg['training']['lr_decay']))
            self.optimizer.lr = new_lr
            # Adjust lambda of Information Gain Reduction Loss
            adjust_info_lambda(it)
            # Print and save results periodically
            print_and_save_results(it, loss_dict)
    def test(self, save_path = None, skip = 8, dsample_ratio=2):
        self.model.eval()
        if self.model_fine is not None:
            self.model_fine.eval()
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        test_id = self.loaded_data['i_split'][2][::skip]
        test_pose = self.loaded_data['poses'][test_id]
        ref_images = self.loaded_data['imgs'][test_id]
        #this should not be an individual function
        # metric = self.test(test_pose, ref_images, not no_metric, not no_metric, False, save_path, dsample_ratio)
        if test_pose.ndim == 2:
            test_pose = test_pose.unsqueeze(0)
        predict = self.__render_image(test_pose, dsample_ratio, save_dir=save_path)
        ref = nn.resize(ref_images, size=(self.img_h // (2 ** dsample_ratio), self.img_w // (2 ** dsample_ratio)), mode='bilinear')
        with jt.no_grad():
            metric = {}
            predict = jt.stack(predict, dim=0)  # [B, 3, H, W]
            if ref is not None:
                psnr_avg = ls.img2psnr_redefine(predict, ref)
                metric['psnr'] = psnr_avg.item()
        print("Metric result is ", metric)
