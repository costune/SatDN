import os
import utm
import json
import rpcm
import torch
import torch.nn.functional as F
import numpy as np
from utils import sat_utils
import matplotlib.pyplot as plt

from icecream import ic


class Dataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        super().__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.train = True
        self.img_downscale = conf.get_float('img_downscale')
        self.data_dir = conf.get_string('data_dir')

        with open(os.path.join(self.data_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.data_dir, json_p) for json_p in json_files if json_p]

        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        all_neighbors = []
        for t, json_p in enumerate(self.json_files):
            # read json, image path and id
            with open(json_p) as f:
                d = json.load(f)
            img_p = os.path.join(self.data_dir, d["img"])
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]

            # get rgb colors
            rgbs = sat_utils.load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            # get rays
            cache_path = "{}/{}.data".format(self.data_dir, img_id)
            if os.path.exists(cache_path) and self.conf.get_bool('force_reload'):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = self.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                torch.save(rays, cache_path)
            
            # get sun direction
            sun_dirs = sat_utils.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]

            # get neighbor rays around piexl
            rays = rays.view(h,w,-1)
            neighbors_per_image = []
            for row in range(h):
                for col in range(w):
                    if row == 0:
                        if col == 0:
                            neighbor_rays = torch.cat([rays[row+1][col],
                                                       rays[row][col+1],
                                                       rays[row][col].repeat(2)], dim=0).reshape(4, -1)
                        elif col == w-1:
                            neighbor_rays = torch.cat([rays[row+1][col],
                                                       rays[row][col-1],
                                                       rays[row][col].repeat(2)], dim=0).reshape(4, -1)
                        else:
                            neighbor_rays = torch.cat([rays[row+1][col],
                                                       rays[row][col+1],
                                                       rays[row][col-1],
                                                       rays[row][col]], dim=0).reshape(4, -1)
                    elif row == h-1:
                        if col == 0:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row][col+1],
                                                       rays[row][col].repeat(2)], dim=0).reshape(4, -1)
                        elif col == w-1:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row][col-1],
                                                       rays[row][col].repeat(2)], dim=0).reshape(4, -1)
                        else:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row][col+1],
                                                       rays[row][col-1],
                                                       rays[row][col]], dim=0).reshape(4, -1)
                    else:
                        if col == 0:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row][col+1],
                                                       rays[row+1][col],
                                                       rays[row][col]], dim=0).reshape(4, -1)
                        elif col == w-1:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row][col-1],
                                                       rays[row+1][col],
                                                       rays[row][col]], dim=0).reshape(4, -1)
                        else:
                            neighbor_rays = torch.cat([rays[row-1][col],
                                                       rays[row+1][col],
                                                       rays[row][col-1],
                                                       rays[row][col+1]], dim=0).reshape(4, -1)
                    neighbors_per_image.append(neighbor_rays)
            all_neighbors += [torch.stack(neighbors_per_image)]

            
            print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
        
        self.all_ids = torch.cat(all_ids, 0).cpu()
        self.all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        self.all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        self.all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        self.all_neighbors = torch.cat(all_neighbors, 0)  # (len(json_files)*h*w, 4, 8)
        self.cal_scaling_params(self.all_rays)
        self.all_rays = self.normalize_rays(self.all_rays)
        self.all_neighbors = self.normalize_rays(self.all_neighbors)
        # ic('max easting: {}, min easting: {}'.format(
        #     torch.max(self.all_rays[:,0]), torch.min(self.all_rays[:,0])))
        # ic('max northing: {}, min northing: {}'.format(
        #     torch.max(self.all_rays[:,1]), torch.min(self.all_rays[:,1])))
        # ic('max height: {}, min height: {}'.format(
        #     torch.max(self.all_rays[:,2]), torch.min(self.all_rays[:,2])))
        # ic('near to far range: {}'.format(torch.min(self.all_rays[:,6]) - torch.max(self.all_rays[:,7])))
        self.all_rays = torch.hstack([self.all_rays, self.all_sun_dirs])  # (len(json_files)*h*w, 11)

        # load depth info
        all_depths = []
        for t, json_p in enumerate(self.json_files):
            with open(json_p) as f:
                d = json.load(f)
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]
            depths_path = os.path.join(self.data_dir, 'depths', '{}.npy'.format(img_id))

            depths = sat_utils.get_depths(depths_path) / self.range

            all_depths += [depths]

            print("Depth {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        self.all_depths = torch.cat(all_depths, 0)  # (len(json_files)*h*w, 1)

        # load depth mask
        all_masks = []
        for t, json_p in enumerate(self.json_files):
            with open(json_p) as f:
                d = json.load(f)
            mask_p = os.path.join(self.data_dir, 'masks', d["img"].replace('RGB', 'CLS'))

            # get mask
            mask = sat_utils.load_mask_from_tiff(mask_p)
            all_masks += [mask]

            print("Mask {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
        
        self.all_masks = torch.cat(all_masks, 0)

        # load cosine infos
        all_consine = []
        for t, json_p in enumerate(self.json_files):
            with open(json_p) as f:
                d = json.load(f)
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]
            cos_path = os.path.join(self.data_dir, 'cosines', '{}.data'.format(img_id))

            cosines = sat_utils.get_normals(cos_path)
            all_consine.append(cosines)

            print("Cosine {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))
        self.all_cosines = torch.cat(all_consine, 0)

        self.val_range = None
        with open(os.path.join(self.data_dir, "{}_DSM.txt".format(img_id[:7])), "r") as f:
            val_data = f.read().split("\n")
        self.val_range = torch.tensor([[float(val_data[0]), float(val_data[1])],
                                       [float(val_data[0]) + float(val_data[2]) * float(val_data[3]), float(val_data[1]) + float(val_data[2]) * float(val_data[3])]]).cpu()
        self.val_range[:, 0] -= self.center[0]
        self.val_range[:, 1] -= self.center[1]
        self.val_range = torch.cat((self.val_range,
                                    torch.tensor([float(d["min_alt"]) - 20.0 - self.center[2],
                                                  float(d["max_alt"]) + 20.0 - self.center[2]]).unsqueeze(-1).cpu()), -1)

        self.val_range[:, 0] /= self.range
        self.val_range[:, 1] /= self.range
        self.val_range[:, 2] /= self.range

        print('Load data: End')

    def cal_scaling_params(self, all_rays):
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_alt_params(all_points[:, 2])
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])]).cpu()
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])])).cpu()

    def normalize_rays(self, rays):
        rays[..., 0] -= self.center[0]
        rays[..., 1] -= self.center[1]
        rays[..., 2] -= self.center[2]
        rays[..., 0] /= self.range
        rays[..., 1] /= self.range
        rays[..., 2] /= self.range
        rays[..., 6] /= self.range
        rays[..., 7] /= self.range
        return rays

    def get_rays(self, cols, rows, rpc, min_alt, max_alt):
        # rpcm.RPCModel.localization
        #TODO 将整个光线采样的距离加长
        min_alts = (float(min_alt) - 50.) * np.ones(cols.shape)
        max_alts = (float(max_alt) + 50.) * np.ones(cols.shape)
        # assume the points of maximum altitude are those closest to the camera
        lons, lats = rpc.localization(cols, rows, max_alts)
        # x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
        # 使用utm时需要根据区域位置改变区域编号
        x_near, y_near, _, _ = utm.from_latlon(lats, lons)
        xyz_near = np.vstack([x_near, y_near, max_alts]).T

        # similarly, the points of minimum altitude are the furthest away from the camera
        lons, lats = rpc.localization(cols, rows, min_alts)
        # x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
        # 同样如上修改
        x_far, y_far, _, _ = utm.from_latlon(lats, lons)
        xyz_far = np.vstack([x_far, y_far, min_alts]).T

        # define the rays origin as the nearest point coordinates
        rays_o = xyz_near

        # define the unit direction vector
        d = xyz_far - xyz_near
        rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

        # assume the nearest points are at distance 0 from the camera
        # the furthest points are at distance Euclidean distance(far - near)
        fars = np.linalg.norm(d, axis=1)
        nears = float(0) * np.ones(fars.shape)

        # create a stack with the rays origin, direction vector and near-far bounds
        rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
        rays = rays.type(torch.FloatTensor)
        return rays
    
    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        if self.train:
            return (self.all_rays[idx], self.all_rgbs[idx], self.all_depths[idx], self.all_masks[idx], self.all_neighbors[idx], self.all_cosines[idx])
        else:
            with open(self.json_files[idx]) as f:
                d = json.load(f)
            return (self.all_rays[torch.where(self.all_ids == idx)[0]], int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale))


if __name__ == '__main__':
    from pyhocon import ConfigFactory

    conf_path = 'confs/sat.conf'
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf_text = conf_text.replace('CASE_NAME', 'JAX_068')
    conf = ConfigFactory.parse_string(conf_text)

    dataset = Dataset(conf['dataset'])
