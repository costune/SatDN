<p align="center">

  <h2 align="center">Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images with Depth and Normal Supervision</h2>
  <p align="center">
    Tianle Liu
    ·
    Shuangming Zhao
    ·
    Wanshou Jiang
    ·
    Bingxuan Guo
  </p>
  <h4 align="center">
    <a href="https://arxiv.org/abs/2502.08352">Paper</a>
  </h4>
</p>

<br>

## News

- **[7 Jul 2025]** The source code of **Sat-DN** is now publicly available!

## 1. Installation

Clone the code and prepare the conda enviroment.

```bash
git clone https://github.com/costune/SatDN.git --recursive
cd SatDN
conda create -n satdn python=3.8
conda activate satdn
```
Install Torch for your CUDA version and the packages in `requirements.txt`.

```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install submodules for the enviroment.

```bash
cd submodule/bundle_adjust
pip install .
```

## 2. Data Preparation

### Build the Original Dataset

We use the [DFC2019](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) to evaluate our method, where `Track 3 / Training data / Reference`, `Track 3 / Training data / RGB images` and `Track 3 / Metadata` are necessary. To obtain the class mask per image, we need the CLS of `JAX_Train.tar` and `OMA_Train.tar` in [Urban Semantic 3D Dataset](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset). Place them in the same folder for subsequent processing. Please refer to the dataset folder structure below.

<details>
  <summary>[folder structure (click to expand)]</summary>

```
  DFC2019/
    ├── Track3-CLS
    │   ├── JAX_004_003_CLS.tif
    │   ├── JAX_004_005_CLS.tif
    │   ├── JAX_004_006_CLS.tif
    │   └── ...
    ├── Track3-Metadata
    │   ├── JAX
    │   │   ├── 01.IMD
    │   │   ├── 01.RPB
    │   │   └── ...
    │   └── OMA
    │       ├── 01.IMD
    │       ├── 01.RPB
    │       └── ...
    ├── Track3-RGB
    │   ├── JAX_004_006_RGB.tif
    │   ├── JAX_004_007_RGB.tif
    │   ├── JAX_004_009_RGB.tif
    │   └── ...
    └── Track3-Truth
        ├── JAX_004_CLS.tif
        ├── JAX_004_DSM.tif
        ├── JAX_004_DSM.txt
        └── ...
```
</details>

### Data Preprocessing

Follow the following steps to preprocess the data. We have also provided a sample data. Place it in the `data` folder to test the code. 

**Preprocess the original dataset**

Copy the data from original dataset, crop the image based on the ROI and run triangulation and bundle adjustment.

```bash
python scripts/DFC2019_Preprocess.py --dfc_dir [DFC2019_DATASET] --out_dir data
```

Modify the `[DFC2019_DATASET]` into your original dataset path.

**Obtain relative depth**

Use Depth-Anything-V2 to obtain relative depth. Please refer to the [docs](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#prepraration) to download the checkpoints. We use `Depth-Anything-V2-Large` as default.

```bash
python scripts/DFC2019_Get_Depth.py --img_dir [SCENE_DIR]
```

Replace `[SCENE_DIR]` with each scene id in the `data` folder.

**Depth fusion**

Fusion between relative depth with sparse 3D points.

```bash
python scripts/DFC2019_Depth_Fusion.py --data_path [SCENE_DIR]
```

Alternatively, refer to `scripts/data_prepare.sh` for convinence.

## 3. Run

```bash
python main.py --case [SCENE_ID] --conf confs/sat.conf
```
The training result is recorded in `results` folder.

## 4. Acknowledgments

Our implementation is built upon [FVMD-ISRe](https://github.com/HEU-super-generalized-remote-sensing/FVMD-ISRe). We also thank the authors of these repositories: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [DepthRegularizedGS](https://github.com/robot0321/DepthRegularizedGS), [MonoSDF](https://github.com/autonomousvision/monosdf).

## 5. Citation

If you think our research is helpful to you, please consider citing it.

```bibtex
@article{liu2025sat,
    title={Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images with Depth and Normal Supervision},
    author={Liu, Tianle and Zhao, Shuangming and Jiang, Wanshou and Guo, Bingxuan},
    journal={arXiv preprint arXiv:2502.08352},
    year={2025}
}
```