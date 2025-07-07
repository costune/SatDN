#/bin/bash

# modify it
DFC2019_ROOT=/dataset/DFC2019
OUTPUT_DIR=data_test
SCENE_ID=('JAX_068')

python scripts/DFC2019_Preprocess.py --dfc_dir "$DFC2019_ROOT" --out_dir "$OUTPUT_DIR" --scene_ids "${SCENE_ID[@]}"

for scene_id in "${SCENE_ID[@]}"; do
    python scripts/DFC2019_Get_Depth.py --img_dir "$OUTPUT_DIR"/"$scene_id"
    python scripts/DFC2019_Depth_Fusion.py --data_path "$OUTPUT_DIR"/"$scene_id"
done