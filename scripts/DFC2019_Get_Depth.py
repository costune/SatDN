import argparse
import cv2
import glob
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.getcwd(), 'submodule', 'Depth-Anything-V2'))

from depth_anything_v2.dpt import DepthAnythingV2

from PIL import Image
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_normals_from_depth(depth_map):
    h, w = depth_map.shape

    dx = np.zeros_like(depth_map)
    dy = np.zeros_like(depth_map)

    dx[:, 1:-1] = (depth_map[:, 2:] - depth_map[:, :-2]) / 2.0  
    dy[1:-1, :] = (depth_map[2:, :] - depth_map[:-2, :]) / 2.0  
    dz = -np.ones_like(depth_map)

    normals = np.zeros((h, w, 3))
    for i in range(1, h-1):
        for j in range(1, w-1):
            tangent_x = np.array([1, 0, dx[i, j]])  
            tangent_y = np.array([0, 1, dy[i, j]])  
            
            normal = np.cross(tangent_x, tangent_y)
            
            normal = normal / np.linalg.norm(normal)
            normals[i, j] = normal

    return normals


def visualize_normals_as_rgb(normals):
    normals_rgb = (normals + 1.0) / 2.0
    normals_rgb = (normals_rgb * 255).astype(np.uint8)
    image = Image.fromarray(normals_rgb, 'RGB')
    return image


def inference(args):
    '''
    Inference depth maps from satellite images using Depth-Anything-V2 model.
    '''

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    out_dir = os.path.join(args.img_dir, 'depth_original')
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'submodule/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    filenames = glob.glob(os.path.join(args.img_dir, '*.json'), recursive=True)
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in tqdm(enumerate(filenames), desc='Depth inference', total=len(filenames)):
        filename = filename.replace('.json', '.tif')
        
        with rasterio.open(filename) as src:
            img_array = src.read()
        img_BGR = cv2.cvtColor(np.moveaxis(img_array, 0, -1), cv2.COLOR_RGB2BGR)
        
        depth = depth_anything.infer_image(img_BGR, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        np.save(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0]),
                depth[..., np.newaxis])

        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((img_BGR.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([img_BGR, split_region, depth])
            
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)


def normal(args):
    '''
    Compute normals from depth maps and save them as images and numpy arrays.
    '''
    depth_folder = os.path.join(args.img_dir, 'depth_original')
    normal_folder = os.path.join(args.img_dir, 'normals')

    os.makedirs(normal_folder, exist_ok=True)
    files = glob.glob(os.path.join(depth_folder, '*.npy'))

    for file in tqdm(files, total=len(files), desc='Compute Normals'):

        depth_map = np.load(file).squeeze()

        normals = compute_normals_from_depth(depth_map)
        normals_rgb_image = visualize_normals_as_rgb(normals)
        normals_rgb_image.save(os.path.join(normal_folder, os.path.basename(file).replace('.npy', '.png')))

        np.save(os.path.join(normal_folder, os.path.basename(file)), normals)


def cosine(args):
    '''
    Compute cosine similarity of normals and save them as images and tensors.
    '''
    normal_path = os.path.join(args.img_dir, 'normals')
    cos_path = normal_path.replace('normals', 'cosines')
    os.makedirs(cos_path, exist_ok=True)

    normal_files = glob.glob(os.path.join(normal_path, "*.npy"))

    for path in tqdm(normal_files, desc='Compute Cosine Similarity', total=len(normal_files)):

        normal_map = np.load(path) 
        img_id = os.path.splitext(os.path.split(path)[-1])[0]

        h, w, _ = normal_map.shape

        similarity_matrix = np.zeros((h, w, 1))

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for i in range(h):
            for j in range(w):
                current_normal = normal_map[i, j]
                
                cosine_sum = 0
                
                for dx, dy in offsets:
                    ni, nj = i + dx, j + dy
                    
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_normal = normal_map[ni, nj]
                    else:
                        neighbor_normal = current_normal
                    
                    dot_product = np.dot(current_normal, neighbor_normal)
                    norm_current = np.linalg.norm(current_normal)
                    norm_neighbor = np.linalg.norm(neighbor_normal)
                    
                    if norm_current != 0 and norm_neighbor != 0:
                        cosine_similarity = dot_product / (norm_current * norm_neighbor)
                    else:
                        cosine_similarity = 1
                    
                    cosine_sum += cosine_similarity
                
                similarity_matrix[i, j] = cosine_sum / 8

        similarity_tensor = torch.tensor(similarity_matrix)

        torch.save(similarity_tensor, os.path.join(cos_path, img_id + '.data'))

        similarity_matrix_normalized = similarity_matrix[:, :, 0]  
        similarity_matrix_normalized = (similarity_matrix_normalized - similarity_matrix_normalized.min()) / \
                                        (similarity_matrix_normalized.max() - similarity_matrix_normalized.min())

        similarity_image = (similarity_matrix_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(cos_path, img_id + '.png'), similarity_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    print("processing {}".format(args.img_dir))
    inference(args)
    normal(args)
    cosine(args)
