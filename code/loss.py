#TODO, info loss will exceed GPU, just delete it
import numpy as np
import jittor as jt
import jittor.nn as nn

# Misc
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.Var([10.]))
def get_psnr(x, y):
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
    ##entrance
    def ray_zvals(self, sigma, acc):
        """
        Compute entropy loss for rays based on sigma values.
        same with the paper
        Args:
            sigma (jt.Var): Sigma values.
            acc (jt.Var): Accumulated values.

        Returns:
            jt.Var: Entropy loss for rays.
        """
        # Compute probability distribution from sigma values
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1) + 1e-10)
        # Compute entropy for the probability distribution
        if self.type_ == 'log2':
            entropy_ray =  -1 * ray_prob * jt.log2(ray_prob + 1e-10)
        elif self.type_ == '1-p':
            entropy_ray =  ray_prob * jt.log2(1 - ray_prob)
        # Sum entropy along the last dimension
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        # Create a mask based on the accumulation values and detach it to prevent gradient flow
        mask = (acc > self.threshold).stop_grad()
        # Apply the mask to the entropy values
        entropy_ray_loss *= mask
        loss_mean = jt.mean(entropy_ray_loss)
        # Apply log scaling if specified, then compute the mean of the entropy loss
        if self.entropy_log_scaling:
            return jt.log(loss_mean + 1e-10)
        return loss_mean