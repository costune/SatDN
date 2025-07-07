from utils import sat_utils
import os
import shutil
import rasterio
import numpy as np

pred_dsm_path = "/home/jiang/ltl/Sat-DN/exp/JAX_167/validations_fine/00100000_dsm.tif"
src_id = "JAX_167"
gt_dir = 'data/JAX_167'
out_dir = "/home/jiang/ltl/Sat-DN/exp/JAX_167/validations_dsm"
epoch_number = "100000"


# evaluate NeRF generated DSM
mae = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, gt_dir, out_dir, epoch_number)
print("Path to output NeRF DSM: {}".format(pred_dsm_path))
print("Altitude MAE: {}".format(np.nanmean(mae)))
rdsm_tmp_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
rdsm_path = rdsm_tmp_path.replace(".tif", "_{:.3f}.tif".format(mae))
shutil.copyfile(rdsm_tmp_path, rdsm_path)
os.remove(rdsm_tmp_path)

# save tmp gt DSM
aoi_id = src_id[:7]
gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
tmp_gt_path = os.path.join(out_dir, "tmp_gt.tif")
if aoi_id in ["JAX_004", "JAX_260"]:
    gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
else:
    gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))
with rasterio.open(gt_seg_path, "r") as f:
    mask = f.read()[0, :, :]
    water_mask = mask.copy()
    water_mask[mask != 9] = 0
    water_mask[mask == 9] = 1
with rasterio.open(rdsm_path, "r") as f:
    profile = f.profile
with rasterio.open(gt_dsm_path, "r") as f:
    gt_dsm = f.read()[0, :, :]
with rasterio.open(tmp_gt_path, 'w', **profile) as dst:
    gt_dsm[water_mask.astype(bool)] = np.nan
    dst.write(gt_dsm, 1)
