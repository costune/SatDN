import os
import torch
import trimesh
import logging
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from shutil import copyfile
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dataset import Dataset
from models.renderer import NeuSRenderer
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, ImplicitNetworkGrid
from icecream import ic


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        self.dataset = Dataset(self.conf['dataset'])
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      num_workers=4,
                                      batch_size=self.conf['train.batch_size'],
                                      pin_memory=True,
                                      generator=torch.Generator(device = 'cuda'))
        self.data_loader_iter = iter(self.data_loader)
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.lr = self.learning_rate
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.depth_weight = self.conf.get_float('train.depth_weight')
        self.smooth_weight = self.conf.get_float('train.smooth_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.epoch = 1

        # Networks
        params_to_train = []
        # self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.sdf_network = ImplicitNetworkGrid(mode=mode, **self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.mlp_parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam([
            {'params': params_to_train, 'lr': self.lr},
            {'params': list(self.sdf_network.grid_parameters()), 'lr': self.lr * 20.0},
        ])

        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        
        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            # 更新renderer中的训练率
            self.sdf_network.get_training_rate(self.end_iter, self.iter_step)

            try:
                rays, true_rgb, true_depth, masks, neighbor_rays, true_cos = next(self.data_loader_iter)
            except StopIteration:
                self.epoch += 1
                self.data_loader_iter = iter(self.data_loader)
                rays, true_rgb, true_depth, masks, neighbor_rays, true_cos = next(self.data_loader_iter)
                            

            rays = rays.to(self.device)
            true_rgb = true_rgb.to(self.device)
            true_depth = true_depth.to(self.device)
            masks = masks.to(self.device)
            neighbor_rays = neighbor_rays.to(self.device)
            # is_surf = is_surf.to(self.device)
            true_cos = true_cos.to(self.device)

            rays_o, rays_d, near, far, sun_dir = rays[:, :3], rays[:, 3: 6], rays[:, 6: 7], rays[:, 7: 8], rays[:, 8: 11]
            
            render_out = self.renderer.render(rays_o, rays_d, near, far, sun_dir,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']  # [batch_size, 3]
            s_val = render_out['s_val']  # [1,]
            cdf_fine = render_out['cdf_fine']  # [batch_size, 128]
            gradient_error = render_out['gradient_error']  # [1,]
            weight_max = render_out['weight_max']  # [batch_size, 1]
            pred_depth = render_out['depth_values']  # [batch_size, 1]

            # Color Loss
            color_error = color_fine - true_rgb
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='mean')
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2).sum() / (color_fine.shape[0] * 3.0)).sqrt())

            # Eikonal Loss
            eikonal_loss = gradient_error

            # Depth Loss
            # depth_loss = self.get_depth_loss(pred_depth, true_depth, mask)
            # if isinstance(depth_loss, float):
            #     depth_loss = torch.tensor(0.0).cuda().float()
            masks = masks.squeeze()
            depth_error = pred_depth[masks] - true_depth[masks]
            depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='mean')

            # Normal Loss
            neighbor_rays = neighbor_rays.view(-1, 8)  # neighbor_rays: [batch_size, 4, 8]
            neighbor_rays_o, neighbor_rays_d, neighbor_near, neighbor_far = neighbor_rays[..., :3], neighbor_rays[..., 3:6], neighbor_rays[..., 6: 7], neighbor_rays[..., 7: 8]
            # use the center depth as around depth
            neighbor_depth = render_out['depth_values'].detach().repeat(1, 4).view(-1, 1)
            center_depth = render_out['depth_values'].detach()
            center_pts = rays_o + center_depth * rays_d
            neighbor_pts = neighbor_rays_o + neighbor_depth * neighbor_rays_d
            # get the noraml of rays
            center_normal = self.sdf_network.gradient(center_pts).squeeze()
            center_normal = center_normal / torch.linalg.norm(center_normal, dim=-1, ord=2, keepdim=True)
            neighbor_normal = self.sdf_network.gradient(neighbor_pts).squeeze()
            neighbor_normal = neighbor_normal / torch.linalg.norm(neighbor_normal, dim=-1, ord=2, keepdim=True)
            center_normal = center_normal.reshape(-1, 1, 3)
            neighbor_normal = neighbor_normal.reshape(-1, 4, 3)
            # cal the mean cosine error
            cos_sim = F.cosine_similarity(center_normal, neighbor_normal, dim=-1).mean(dim=1)
            normal_loss = F.mse_loss(cos_sim, true_cos, reduction='mean')
            
            # Total Loss
            loss = color_fine_loss + \
                    eikonal_loss * self.igr_weight + \
                    depth_loss * self.depth_weight + \
                    normal_loss * self.smooth_weight

            # ic(color_fine_loss.detach())
            # ic(eikonal_loss.detach() * self.igr_weight)
            # ic(depth_loss.detach() * self.depth_weight)
            # ic(normal_loss.detach() * self.smooth_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/depth_loss', depth_loss, self.iter_step)
            self.writer.add_scalar('Loss/normal_loss', normal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/lr', self.lr, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val, self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1]).mean(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max).mean(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            self.writer.add_scalar('Statistics/epoch', self.epoch, self.iter_step)

            self.update_learning_rate()

            if self.iter_step != 0 and self.iter_step % 20000 == 0:
                self.save_checkpoint()
                self.validate_image()
                self.validate_mesh()

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        self.lr = self.learning_rate * learning_factor
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.epoch = checkpoint['epoch']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'epoch': self.epoch
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(len(self.dataset.json_files))

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        self.dataset.train = False
        rays, H, W = self.dataset[idx]
        self.dataset.train = True
        rays_o, rays_d, near, far, sun_dirs = rays[:, :3].cuda(), rays[:, 3:6].cuda(), rays[:, 6:7].cuda(), rays[:, 7:8].cuda(), rays[:, 8:].cuda()
        rays_o = rays_o.split(self.batch_size)
        rays_d = rays_d.split(self.batch_size)
        near = near.split(self.batch_size)
        far = far.split(self.batch_size)
        sun_dirs = sun_dirs.split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, near_batch, far_batch, sun_dirs_batch in zip(rays_o, rays_d, near, far, sun_dirs):
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              sun_dirs_batch,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = (np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_rgb.png'.format(self.iter_step, i, idx)),
                           img_fine[..., i][:, :, [2, 1, 0]])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])


    def validate_mesh(self, resolution=512, threshold=0.0):
        u, vertices, triangles = self.renderer.extract_geometry(self.dataset.val_range[0], self.dataset.val_range[1], resolution=resolution + 1, threshold=threshold)
        vertices *= self.dataset.range.numpy()
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        u = np.flip(u, axis=2)
        sdf_0 = np.zeros((resolution + 1, resolution + 1), dtype=np.float32)
        dsm = np.zeros((resolution, resolution), dtype=np.float32)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                for k in range(u.shape[2]):
                    if u[i, j, k] >= 0.0:
                        pre = u[i, j, k-1]
                        post = u[i, j, k]
                        sdf_0[i, j] = k + (pre / (pre - post))
                        break
        
        for i in range(resolution):
            for j in range(resolution):
                dsm[i, j] = (sdf_0[i, j] + sdf_0[i + 1, j] + sdf_0[i, j + 1] + sdf_0[i + 1, j + 1]) * 0.25
        # dsm = sdf_0[:-1, :-1]
        dsm = dsm / resolution
        dsm = self.dataset.val_range[1, 2].numpy() - dsm * (self.dataset.val_range[1, 2].numpy() - self.dataset.val_range[0, 2].numpy())
        dsm = dsm * self.dataset.range.numpy() + self.dataset.center[2].numpy()
        dsm = np.rot90(dsm, 1)
        cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_dsm.tif'.format(self.iter_step)), dsm)

        logging.info('End')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/sat.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='OMA_203')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(resolution=512, threshold=args.mcube_threshold)
