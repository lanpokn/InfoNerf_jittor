##not rewrite, but need understand
#if you rewrite it, previous trains can't be use anymore
import jittor as jt
import jittor.nn as nn
from utils import *
class NeRF_network(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        NeRF MLP 
        Args:
            D: number of layers of MLP
            W: dimension (depth) of MLP
            input_ch: number of channels of input 
            input_ch_views: number of channels of direction map
            output_ch: output channels (usually rgb + sigma)
            skips: input skip-connection
            use_viewdirs: use view direction as an additional condition of RGB 
        """
        super(NeRF_network, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.Sequential(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.Sequential([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, x):
        if not self.use_viewdirs and x.shape[-1] == self.input_ch:
            input_pts = x
        else:
            input_pts = x[..., :self.input_ch]
            input_views = x[..., self.input_ch:]
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for _layer in self.views_linears:
                h = _layer(h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs 
def create_nerf(cfg):
    cfg_ren = cfg['rendering']
    cfg_train = cfg['training']
    embed_fn, input_ch = get_embedder(cfg_ren['multires'], cfg_ren['i_embed'])

    input_ch_views = 0
    embeddirs_fn = None
    if cfg['rendering']['use_viewdirs']:
        embeddirs_fn, input_ch_views = get_embedder(cfg_ren['multires_views'], cfg_ren['i_embed'])
    output_ch = 5 if cfg_ren['N_importance'] > 0 else 4
    
    model = NeRF_network(
        D=cfg_train['netdepth'], W=cfg_train['netwidth'],
        input_ch=input_ch, output_ch=output_ch, skips=[4, ],
        input_ch_views=input_ch_views, use_viewdirs=cfg['rendering']['use_viewdirs'])

    model_fine = None
    if cfg_ren['N_importance'] > 0:
        model_fine = NeRF_network(
            D=cfg_train['netdepth_fine'], W=cfg_train['netwidth_fine'],
            input_ch=input_ch, output_ch=output_ch, skips=[4, ],
            input_ch_views=input_ch_views, use_viewdirs=cfg['rendering']['use_viewdirs'])

    return model, model_fine, embed_fn, embeddirs_fn

def run_network(inputs, viewdirs, model, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Run NeRF model (embedding + MLP)
    inputs: [..., 3]
    viewdirs: [..., 3]
    """
    use_dir = (viewdirs is not None) and (embeddirs_fn is not None)
    input_shape = inputs.shape
    point_num = 1
    for _d in range(len(input_shape) - 1): point_num *= input_shape[_d]
    group_size = netchunk
    group_num = (
        (point_num // group_size) if (point_num % group_size == 0) else (point_num // group_size + 1))
    if group_num == 0:
        group_num = 1
        group_size = point_num

    pt_group_output = []
    inputs = inputs.reshape([-1, input_shape[-1]])  # [N, 3], flatten points
    if use_dir:
        viewdirs = viewdirs.reshape([-1, viewdirs.shape[-1]])  # [N, 3]
    for gi in range(group_num):
        start = gi * group_size
        end = (gi + 1) * group_size
        end = (end if (end <= point_num) else point_num)

        pt_group = inputs[start:end, :]  # [group_p_num, 3]
        embedded = embed_fn(pt_group)  # [group_p_num, ch]

        if use_dir:
            dir_group = viewdirs[start:end, :]  # [group_p_num, 3]
            embedded_dirs = embeddirs_fn(dir_group)  # [group_p_num, ch]
            embedded = jt.concat([embedded, embedded_dirs], dim=-1)
        pt_group_output.append(model(embedded))  # [group_p_num, out_ch]
    
    output_flat = jt.concat(pt_group_output, dim=0)  # [N, , out_ch]
    return jt.reshape(output_flat, list(input_shape[:-1]) + [output_flat.shape[-1], ])
