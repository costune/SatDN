import copy
import torch
import rasterio
import numpy as np
from PIL import Image
from torchvision import transforms as T

def get_depths(data_path):
    depths = np.load(data_path)
    depths = torch.from_numpy(depths).view(-1, 1)  # (h*w, 1)
    depths = depths.type(torch.FloatTensor)

    return depths

def get_normals(data_path):
    normals = torch.load(data_path).flatten()
    normals = normals.type(torch.FloatTensor)

    return normals

def load_mask_from_tiff(mask_path):
    with rasterio.open(mask_path, "r") as f:
        mask = f.read()[0, :, :]
        water_mask = mask.copy()
        water_mask[mask != 9] = True
        water_mask[mask == 9] = False

    water_mask = torch.from_numpy(water_mask).view(-1, 1)
    water_mask = water_mask.type(torch.BoolTensor)
    
    return water_mask

def load_tensor_from_rgb_geotiff(img_path, downscale_factor=1.0, imethod=Image.BICUBIC):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs

def rescale_rpc(rpc, alpha):
    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled

def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    scale = (v.max() - v.min()) / 2
    offset = v.min() + scale
    return scale, offset

def rpc_scaling_alt_params(v):
    """
    find the scale and offset of a vector
    """
    scale = (v.max() - v.min()) / 2
    offset = v.mean()
    return scale, offset

def latlon_to_ecef_custom(lat, lon, alt):
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def get_sun_dirs(sun_elevation_deg, sun_azimuth_deg, n_rays):
    """
    Get sun direction vectors
    Args:
        sun_elevation_deg: float, sun elevation in  degrees
        sun_azimuth_deg: float, sun azimuth in degrees
        n_rays: number of rays affected by the same sun direction
    Returns:
        sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
    """
    sun_el = np.radians(sun_elevation_deg)
    sun_az = np.radians(sun_azimuth_deg)
    sun_d = -1 * np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
    sun_d = sun_d / np.linalg.norm(sun_d)
    sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
    sun_dirs = sun_dirs.type(torch.FloatTensor)
    return sun_dirs

def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt

import datetime
import os
import shutil

def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    # from osgeo import gdal

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    shutil.copyfile(in_dsm_path, pred_dsm_path)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    # xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    # xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    # resolution = dsm_metadata[3]

    # define projwin for gdal translate
    # ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    # ds = gdal.Open(in_dsm_path)
    # ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    # ds = None
    # os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, source_path, crop_path))
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    try:
        import dsmr
    except:
        print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
        fix_xy = True
    if fix_xy:
        pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        import dsmr
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read()[0, :, :]
    err = pred_rdsm - gt_dsm
    
    med = np.nanmedian(np.abs(err.flatten()))
    prec_1 = np.sum(np.abs(err.flatten()) <= 1.0) / len(err.flatten())
    print("MED: {}".format(med))
    print("Prec-1: {}".format(prec_1))

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)

    return err

def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, gt_dir, out_dir, epoch_number, save=True):
    # save dsm errs
    aoi_id = src_id[:7]
    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
    gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))
    assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"
    gt_roi_metadata = np.loadtxt(gt_roi_path)
    rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff_epoch{}.tif".format(src_id, epoch_number))
    rdsm_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                                       out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)
    #os.system(f"rm tmp*.tif.xml")
    np.savetxt('error.txt', diff)
    if not save:
        os.remove(rdsm_diff_path)
        os.remove(rdsm_path)
    return np.nanmean(abs(diff.ravel()))