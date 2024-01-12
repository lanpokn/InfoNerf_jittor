#TODO, info loss will exceed GPU, just delete it
import numpy as np
import jittor as jt
import jittor.nn as nn

# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.Var([10.]))
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def img2psnr_redefine(x, y):
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)
    
    [new]
    average PSNR(each image pair)
    '''
    image_num = x.size(0)
    mses = ((x - y) ** 2).reshape(image_num, -1).mean(-1)
    
    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.stack(psnrs).mean()
    return psnr

# def img2psnr_mask(x, y, mask):
#     '''
#     we redefine the PSNR function,
#     [previous]
#     average MSE -> PSNR(average MSE)
    
#     [new]
#     average PSNR(each image pair)
#     '''
#     image_num = x.size(0)
#     mses = ((x-y)**2).mean(-1)
#     mses_sum = (mses*mask).reshape(image_num, -1).sum(-1)
#     mses = mses_sum /mask.reshape(image_num, -1).sum(-1)
#     psnrs = [mse2psnr(mse) for mse in mses]
#     psnr = jt.mean(jt.stack(psnrs))
#     return psnr

#Ray Entropy Minimization Loss
class EntropyLoss:
    def __init__(self, args):
        super(EntropyLoss, self).__init__()
        self.type_ = args['entropy_type'] 
        self.threshold = args['entropy_acc_threshold']
        self.computing_entropy_all = args['computing_entropy_all']
        self.computing_ignore_smoothing = args['computing_entropy_all']
        self.entropy_log_scaling = args['computing_entropy_all']
        self.N_entropy = args['N_entropy'] 
        
        if self.N_entropy == 0:
            self.computing_entropy_all = True
    
    def ray(self, density, acc):
        density = nn.relu(density[..., -1])
        sigma = 1 - jt.exp(-density)
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1) + 1e-10)
        entropy_ray = jt.sum(self.entropy(ray_prob), -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray*= mask
        entropy_ray_loss = jt.mean(entropy_ray, -1)
        if self.entropy_log_scaling:
            return jt.log(entropy_ray_loss + 1e-10)
        return entropy_ray_loss

    def ray_zvals(self, sigma, acc):
        ray_prob = sigma / (jt.sum(sigma,-1).unsqueeze(-1) + 1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc > self.threshold).stop_grad()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)
    
    def ray_zvals_ver1_sigma(self, sigma, dists, acc):
        ray_prob = sigma / (jt.sum(sigma* dists,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        
        #intergral
        entropy_ray = entropy_ray * dists
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def ray_zvals_ver2_alpha(self, alpha, dists, acc):
        ray_prob = alpha / (jt.sum(alpha,-1).unsqueeze(-1)+1e-10)
        
        entropy_ray = -1 * ray_prob * jt.log2(ray_prob/(dists+1e-10)+1e-10)
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)
    
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1 * prob * jt.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob * jt.log2(1-prob)
