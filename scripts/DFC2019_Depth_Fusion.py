import os
import cv2
import json
import glob
import rpcm
import utm
import numpy as np
import torch
import argparse
import sys
sys.path.append(os.path.join(os.getcwd()))
from utils.sat_utils import ecef_to_latlon_custom
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import logging


def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.05):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and(torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths not in [0,1]
        # loss_hinge1 = loss_hinge2 = 0.0
        # if (source_hat<=0.0).any():
        #     loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        # loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            logging.info(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")
    logging.info(f"loss ={loss:10.5f}\n")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss

def load_keypoint_weights(json_files, points):
    '''
    calculate the weights from reprojection errors
    '''
    n_pts = points.shape[0]
    n_cams = len(json_files)
    reprojection_error = np.zeros((n_pts, n_cams), dtype=np.float32)
    for t, json_p in enumerate(json_files):
        with open(json_p) as f:
            d = json.load(f)

        if "keypoints" not in d.keys():
            raise ValueError("No 'keypoints' field was found in {}".format(json_p))
        
        pts2d = np.array(d["keypoints"]["2d_coordinates"])
        pts3d = np.array(points[d["keypoints"]["pts3d_indices"], :])

        rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")

        lat, lon, alt = ecef_to_latlon_custom(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
        col, row = rpc.projection(lon, lat, alt)
        pts2d_reprojected = np.stack((col, row), axis=-1)
        errs_obs_current_cam = np.linalg.norm(pts2d - pts2d_reprojected, axis=1)
        reprojection_error[d['keypoints']['pts3d_indices'], t] = errs_obs_current_cam

    # 将所有点视为整体进行获得归一化权重，这是合理的
    e = np.sum(reprojection_error, axis=1)
    e_mean = np.mean(e)
    weights = np.exp(-(e / e_mean) ** 2)

    return weights

def read_depth(data_path):
    '''
    load relative images depth and sparse points depth
    '''
    print('Loading sparse and dense depth...')
    print()

    # load the image and dense depth file names
    json_files = glob.glob(os.path.join(data_path, '*.json'))
    depth_files = glob.glob(os.path.join(data_path, 'depth_original', '*.npy'))
    
    # check file names match
    for json_file in json_files:
        file_name = os.path.splitext(os.path.basename(json_file))[0] + '.npy'
        if not os.path.join(data_path, 'depth_original', file_name) in depth_files:
            raise FileNotFoundError("The depth data is not consistent with the tiff data")
    
    # check the sparse points file
    if os.path.exists(os.path.join(data_path, 'pts3d.npy')):
        pcd = np.load(os.path.join(data_path, 'pts3d.npy'))
    else:
        raise FileNotFoundError("Could not find {}".format(os.path.join(data_path, 'pts3d.npy')))
    
    # get per point weight
    weights = load_keypoint_weights(json_files, pcd)
    
    # convert the coord of sparse points from Geocentric Coordinates into lat-lon-alt
    lat_all, lon_all, alt_all = ecef_to_latlon_custom(pcd[:, 0], pcd[:, 1], pcd[:, 2])
    pcd = np.stack([lat_all, lon_all, alt_all], axis=-1)

    # load dense depth and sparse depth
    depth_infos = {}
    for i, json_file in enumerate(json_files):
        with open(json_file) as f:
            d = json.load(f)
        img_id = d['img'].split('.')[0]

        # sparse info for each image
        rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
        pts2d = np.round(np.array(d['keypoints']['2d_coordinates'])).astype(np.int32)  # [col, row]
        pts3d = pcd[d['keypoints']['pts3d_indices']]  # [easting, northing, alt]
        weight = weights[d['keypoints']['pts3d_indices']]
        min_alts = (float(d['min_alt']) - 50.) * np.ones(len(pts2d))
        max_alts = (float(d['max_alt']) + 50.) * np.ones(len(pts2d))

        # Use RPC to parameterize reference plane
        lons, lats = rpc.localization(pts2d[:, 0], pts2d[:, 1], max_alts)
        x_near, y_near, _, _ = utm.from_latlon(lats, lons)
        xyz_near = np.vstack([x_near, y_near, max_alts]).T

        lons, lats = rpc.localization(pts2d[:, 0], pts2d[:, 1], min_alts)
        x_far, y_far, _, _ = utm.from_latlon(lats, lons)
        xyz_far = np.vstack([x_far, y_far, min_alts]).T

        # surface position
        surface_alts = pts3d[:, 2]
        lons, lats = rpc.localization(pts2d[:, 0], pts2d[:, 1], surface_alts)
        x_surface, y_surface, _, _ = utm.from_latlon(lats, lons)
        xyz_surface = np.vstack([x_surface, y_surface, surface_alts]).T

        # sparse relative depth with respect to near plane
        depth_sparse = np.linalg.norm((xyz_surface - xyz_near), axis=-1)
        whole_depth = np.linalg.norm((xyz_far - xyz_near), axis=-1)
        
        # dense info for each image
        # convert dense depth for better optim
        depth_dense = 1. - np.load(os.path.join(data_path, 'depth_original', img_id + '.npy')) / 255.0
        depth_dense = depth_dense.squeeze()
        # plt.imshow(depth_dense)
        # plt.colorbar()
        # plt.savefig('debug.png')

        # store per image depth info in a dict
        depth_infos[img_id] = [d, pts2d, depth_sparse, weight, depth_dense]  # [meta data, 2d proj coords of 3d points, sparse depth map, weight, dense depth map]

    print('Done')
    print()
    return depth_infos


def make_depth_dataset(args, depth_infos: dict):
    '''
    fir enterence for every image
    '''
    for img_id, data in depth_infos.items():
        print('Optimizing {} depth...'.format(img_id))
        logging.info(img_id)
        d, pts2d, sparse, weight, depth_dense = data
        height, width = d['height'], d['width']
        
        # use sparse depth to fill the dense
        depth_sparse, depth_weight = np.zeros((height, width)), np.zeros((height, width))
        depth_sparse[pts2d[:, 1], pts2d[:, 0]] = sparse
        # for i in range(height):
        #     for j in range(width):
        #         if depth_sparse[i, j] != 0.:
        #             depth_dense[i, j] = 1.
        # plt.imshow(depth_dense, cmap='Greys_r')
        # plt.axis(False)
        # plt.savefig('debug_{}_sparse.png'.format(img_id))
        depth_weight[pts2d[:, 1], pts2d[:, 0]] = weight

        # target = depth_sparse.copy()
        # target = ((target != 0) * 255).astype(np.uint8)
        # plt.imshow(target)
        # plt.axis(False)
        # plt.savefig('debug_{}_sparse.png'.format(img_id))

        # optimization
        depthmap, depthloss = optimize_depth(source=depth_dense, target=depth_sparse, mask=depth_weight>0.0, depth_weight=depth_weight)

        # save as image
        output_file = os.path.join(args.data_path, 'depths', '{}.png'.format(img_id))
        plt.figure()
        plt.imshow(depthmap)
        plt.axis(False)
        plt.colorbar(label='range(m)')
        plt.savefig(output_file, dpi=300)
        plt.close()

        # save as array
        output_file = output_file.replace('.png', '.npy')
        np.save(output_file, depthmap)

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/JAX_train/JAX_068')
    parser.add_argument('--cpu', action='store_true', default='if set, using cpu')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_path, 'depths'), exist_ok=True)
    # 配置日志输出到文件
    logging.basicConfig(filename=os.path.join(args.data_path, 'depths', 'recording.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


    depth_infos = read_depth(args.data_path)
    make_depth_dataset(args, depth_infos)
    print('Optimization done.')
    logging.info('Done.\n')
