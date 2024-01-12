##this file should and can be rewrite
import jittor as jt
import jittor.nn as nn
import numpy as np
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], dim=-1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

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