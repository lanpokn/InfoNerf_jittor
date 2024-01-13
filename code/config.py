import toml
import os
import numpy as np
import json
import cv2 as cv
import jittor as jt

def load_config(cfg_path, base_path=None):
    """
    Load configuration from a TOML file.

    Args:
        cfg_path (str): Path to the main configuration file.
        base_path (str, optional): Path to the base configuration file. Default is None.

    Returns:
        dict: Merged configuration dictionary.
    """
    cfg = {}

    # If no base_path is provided, directly load the cfg_path
    if base_path is None:
        return toml.load(cfg_path)

    # Load base configuration from base_path
    cfg = toml.load(base_path)

    # Load additional configuration from cfg_path
    cfg_ex = toml.load(cfg_path)

    # Merge the base configuration and additional configuration
    return merge_dict(cfg, cfg_ex)

def merge_dict(dict1, dict2):
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary.
    """
    # Create a copy of dict1 to avoid modifying the original dictionary
    result = dict(dict1)

    # Iterate through items in dict2
    for key in dict2:
        # If the key is already present in result
        if key in result:
            # If both values are dictionaries, recursively merge them
            if type(result[key]) is dict and type(dict2[key]) is dict:
                result[key] = merge_dict(result[key], dict2[key])
            else:
                # Otherwise, update the value in result with the value from dict2
                result[key] = dict2[key]
        else:
            # If the key is not present in result, add it with its value from dict2
            result[key] = dict2[key]

    # Return the merged dictionary
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

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    Load Blender dataset from specified directory.

    Args:
        basedir (str): Base directory containing dataset files.
        half_res (bool, optional): Flag to load half-resolution images. Default is False.
        testskip (int, optional): Skip factor for test split. Default is 1.

    Returns:
        dict: Loaded data including images, poses, render poses, calibration information, and split indices.
    """
    # Define data splits
    splits = ['train', 'val', 'test']
    metas = {}

    # Load metadata for each split
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    # Iterate through splits and load images and poses
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        # Set skip factor based on split and testskip value
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # Load images and poses for each frame
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(cv.imread(fname, cv.IMREAD_UNCHANGED))
            poses.append(np.array(frame['transform_matrix']))

        # Normalize image values to range [0, 1]
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [list(np.arange(counts[i], counts[i + 1])) for i in range(3)]

    # Concatenate images and poses across all splits
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # Extract calibration parameters
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Generate render poses
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # Adjust resolution and focal length if half_res is True
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        imgs = imgs_half_res

    # Cast intrinsics to the right types
    H, W = int(H), int(W)

    # Create dictionary with loaded data
    loaded_data = {
        'imgs': imgs,
        'poses': poses,
        'render_poses': render_poses,
        'calib': [H, W, focal],
        'i_split': i_split
    }

    return loaded_data


def load_blender_data_ex(args):
    """
    Load Blender dataset with additional options.

    Args:
        args (dict): Dictionary containing configuration options.

    Returns:
        dict: Loaded data including images, poses, render poses, calibration information, and split indices.
    """
    # Check if 'datadir' is present in args
    assert 'datadir' in args

    # Check whether to load from cache or not
    load_from_cache = ('load_cache' not in args) or (not args['load_cache'])
    
    # If cache file exists and load_from_cache is True, load data from cache
    if os.path.exists(os.path.join(args['datadir'], 'cache.bin')) and load_from_cache:
        print("Load from cache.")
        raw_data = jt.load(os.path.join(args['datadir'], 'cache.bin'))
        
        # Convert numpy arrays to jittor.Var
        for k in raw_data.keys():
            if isinstance(raw_data[k], np.ndarray):
                raw_data[k] = jt.array(raw_data[k])
        
        return raw_data

    # Prepare arguments for load_blender_data function
    load_arg = {'basedir': args['datadir']}
    if 'half_res' in args:
        load_arg['half_res'] = args['half_res']
    if 'testskip' in args:
        load_arg['testskip'] = args['testskip']

    # Load Blender data using load_blender_data function
    raw_data = load_blender_data(**load_arg)
    imgs = raw_data['imgs']

    # Adjust images based on background color option
    if 'white_bkgd' in args and args['white_bkgd']:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]
    
    raw_data['imgs'] = imgs

    # Extract split indices
    i_train, i_val, i_test = raw_data['i_split']

    # Few-shot training option
    if 'fewshot' in args and args['fewshot'] > 0:
        if 'train_scene' in args:
            i_train = args['train_scene']
        else:
            np.random.seed(args['fewshot_seed'])
            i_train = np.random.choice(i_train, args['fewshot'], replace=False)
        print('Few-shot training ', i_train)
        raw_data['i_split'] = [i_train, i_val, i_test]

    # Convert numpy arrays to jittor.Var
    for k in raw_data.keys():
        if isinstance(raw_data[k], np.ndarray):
            raw_data[k] = jt.array(raw_data[k])
    
    # Permute image dimensions to match the expected format
    raw_data['imgs'] = raw_data['imgs'].permute(0, 3, 1, 2)

    return raw_data
