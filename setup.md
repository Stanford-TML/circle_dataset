## Setup Instructions

1. **Download Project Files**
- [Scence + Subject URDF Files (Circle Assets)](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_assets.zip)
- [Motion Trajectories (Circle Movement)](https://circledataset.s3.us-west-2.amazonaws.com/release/CIRCLE_movement.zip)

2. **Create a conda environment**

```bash
conda create -n circle_dataset python=3.9 -y
conda activate circle_dataset
```

3. **Install Habitat Sim**

```bash
pip install cmake
conda install habitat-sim -c conda-forge -c aihabitat -y
```

4. **Install other dependencies**

```bash
pip install fairmotion --no-deps
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
pip install black
pip install dataclasses
pip install git+https://github.com/nghorbani/human_body_prior.git@0278cb45180992e4d39ba1a11601f5ecc53ee148
pip install matplotlib
pip install PyOpenGL==3.1.0
pip install pyrender==0.1.39
pip install scikit-learn
pip install tqdm
pip install git+https://github.com/nghorbani/body_visualizer.git@be9cf756f8d1daed870d4c7ad1aa5cc3478a546c
```

5. **Run circle viewer**

```bash
python circle_viewer.py \ 
    --scene 102815835 \
    --dataset <path_to_circle_assets_folder>/floorplanner.scene_dataset_config.json \
    --experiment_dir <path_to_circle_movement_folder>/s1/bathroom1_lights_1_bathroom1/0/001_reaching \
    --urdf <path_to_circle_assets_folder>/subjects/1.urdf
```