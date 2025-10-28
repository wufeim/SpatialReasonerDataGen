# Installation

Please check INSTALL.md for installation instructions.

Follow the instructions to install dependencies. See [Troubleshooting](#troubleshooting) for known issues.

1. Install basic libraries.

    ```sh
    conda create -n srdatagen python=3.10
    conda activate srdatagen

    # Install PyTorch
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

    # Install mmengine
    pip install -U openmim
    mim install mmengine

    # Other libraries
    pip install iopath pyequilib==0.3.0 albumentations einops open3d imageio yacs transforms3d
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
    ```

2. Install `Grounded-Segment-Anything`.

    ```sh
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    cd Grounded-Segment-Anything/
    python -m pip install -e segment_anything
    pip install --no-build-isolation -e GroundingDINO
    ```

3. Install `SAM2`. Update `./sam-hq/sam-hq2/setup.py` and `./sam-hq/sam-hq2/pyproject.toml` accordingly so it does not overwrite your PyTorch setup.

    ```sh
    git clone https://github.com/SysCV/sam-hq.git
    cd sam-hq/sam-hq2
    pip install -e .
    ```

4. Install `Depth-Anything-V2`.

    ```sh
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    ```

5. Install `recognize-anything`.

    ```sh
    git clone https://github.com/xinyu1205/recognize-anything.git
    pip install -r ./recognize-anything/requirements.txt
    pip install setuptools --upgrade
    pip install -e ./recognize-anything/
    ```

6. Install `PerspectiveFields`.

    ```sh
    git clone https://github.com/jinlinyi/PerspectiveFields.git
    ```

7. Download pretrained weights.

    ```sh
    wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth -P pretrained_weights
    wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth -P pretrained_weights
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P pretrained_weights
    wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt -P pretrained_weights
    wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/croplargeEX2/dino_weight.pt -P pretrained_weights
    ```

## Troubleshooting

1. Getting `GlobalHydra` not initialized error when building `SAM2`:

    ```sh
    AssertionError: GlobalHydra is not initialized, use @hydra.main() or call one of the hydra initialization methods first
    ```

    Update the `build_sam2` function in `sam-hq/sam-hq2/sam2/build_sam.py` with `hydra.initialize`:

    ```py
    def build_sam2(...):
      ...
      # Read config and init model
      from hydra import initialize
      with initialize(version_base=None, config_path='configs', job_name='srdatagen'):
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
      ...
    ```

    Also update the config:

    ```py
    sam_cfg.cfg_path = 'sam2.1/sam2.1_hq_hiera_l.yaml'
    ```
