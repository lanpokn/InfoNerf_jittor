import toml
import os

def load_config(cfg_path, base_path=None):
    cfg = {}
    if base_path is None:
        return toml.load(cfg_path)
    cfg = toml.load(base_path)
    cfg_ex = toml.load(cfg_path)
    return merge_dict(cfg, cfg_ex)


def merge_dict(dict1, dict2):
    result = dict(dict1)
    for i in dict2:
        if i in result:
            if type(result[i]) is dict and type(dict2[i]) is dict:
                result[i] = merge_dict(result[i], dict2[i])
            else:
                result[i] = dict2[i]
        else:
            result[i] = dict2[i]
    return result

def set_config(cfg_path, base_path,ckptpath):
    Cfg = load_config(cfg_path, base_path)
    Cfg["training"]["ckpt"] = ckptpath
    return Cfg

def prepare_dir(cfg):
    # Create log dir and copy the config file
    basedir = cfg['basedir']
    expname = cfg['expname']
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'result'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.toml')
    with open(f, 'w') as _f:
        toml.dump(cfg, _f)
    return os.path.join(basedir, expname)