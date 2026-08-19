# XSIM: Unified Sensor Simulation for Autonomous Driving

**Official implementation of the IJCAI 2026 paper**
["Unified Sensor Simulation for Autonomous Driving"](https://arxiv.org/abs/2602.05617)

Nikolay Patakin, Arsenii Shirokov, Anton Konushin, Dmitry Senushkin

[![arXiv](https://img.shields.io/badge/arXiv-2602.05617-b31b1b.svg)](https://arxiv.org/abs/2602.05617)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](#installation)

XSIM rebuilds a recorded driving log into an editable, re-renderable 3D scene: a static background plus explicit, 
time-parameterized dynamic objects, each carrying its own set of Gaussians along its trajectory. 
Feed it a multi-camera + multi-LiDAR sequence, and get back a model you can render from novel views and at novel times — 
the building block for closed-loop simulation, sensor re-simulation, and scenario editing.

## Installation

#### System requirements:

- CUDA 12 (12.6 and 12.8 tested) or CUDA 13.x toolkit
- GCC/G++ with C++17 support

To run training you need a GPU with 24GB or more VRAM, to run inference — 12GB or more.

#### 0. Clone the repository

Clone with submodules:

```bash
git clone --recursive https://github.com/whesense/xsim
```

If you already cloned without `--recursive`, pull them in after the fact:

```bash
git submodule update --init --recursive
```

To reconstruct or render scenes with humans, you need the SMPL body model: register at
[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/), download the official SMPL release (version 1.0.0,
`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) and convert it into the format XSIM loads:

```bash
python tools/data/convert_smpl_model.py path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

This writes `smpl_models/SMPL_NEUTRAL.pth` (override with `--output`); alternatively point the `SMPL_PATH`
environment variable at wherever you saved it. Note that the SMPL body model is licensed by the
Max Planck Institute for non-commercial scientific research — see the [licensing section](#license) below.

#### 1. Create the conda environment

Tested with Python 3.11–3.12. Generally any preexisting environment with a recent 2.x PyTorch installation should work, 
but it's better to create a separate environment:

```bash
conda create -n xsim python=3.12
conda activate xsim
```

#### 2. Install PyTorch

Match the build to your CUDA version if pip doesn't pick it up automatically:

```bash
# add " --index-url https://download.pytorch.org/whl/cuXXX" for your CUDA toolkit version, e.g. cu132 or cu128
pip install torch torchvision
```

#### 3. Install pytoast/xsimgs

Install `pytoast` before `xsimgs`, don't merge them into a single command.
`--no-build-isolation` is required — the build needs the PyTorch you just installed, and isolation hides it. Don't drop the flag.

`xsimgs` needs `pybind11` at build time, and `--no-build-isolation` won't pull it in, so install it first:

```bash
pip install "pybind11>=3.0.4"
pip install --no-build-isolation third_party/pytoast
pip install --no-build-isolation third_party/xsimgs
```

#### 4. Install the remaining requirements

Some of these build against PyTorch too, so keep `--no-build-isolation` here as well:

```bash
pip install --no-build-isolation -r requirements.txt
```

Optionally, you can also install `gsplat` (required for the MCMC strategy and if you want to cross-check against it):

```bash
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat@v1.5.3
```

#### 5. Install XSIM

```bash
pip install -e .
```

Editable install, so `import xsim` points straight at `src/xsim/` and your edits take effect without reinstalling.

## Data setup

Machine-local dataset locations live in `configs/data/dataset_paths.yaml`, which is gitignored so every machine keeps its own copy.
Create it from the template and fill in the paths for the datasets you use:

```bash
cp configs/data/.dataset_paths.template.yaml configs/data/dataset_paths.yaml
```

An exported environment variable always overrides the file, so one-off runs can just do 
`WAYMO_ROOT_PATH=/other/waymo python tools/train.py ...`.

Per-dataset download and path instructions live in `docs/datasets/`:

- [Waymo Open Dataset](docs/datasets/waymo.md)

## Quickstart

Train a scene:

```bash
python tools/train.py configs/waymo_default.yaml
```

Any config value can be overridden from the command line as `dotted.key=value`:

```bash
python tools/train.py configs/waymo_default.yaml provider.scene_idx=555
```

Each run creates an experiment directory under `outputs/` with the resolved config, checkpoints (`nodes_XXXXXX.pth`), 
periodic visualizations, and `metrics.txt`.

Evaluate a trained experiment (PSNR / SSIM / LPIPS on the train and test splits, Chamfer distance for LiDAR):

```bash
python tools/eval.py outputs/<experiment_dir>
```

Render every frame next to its ground truth (images per camera, point clouds per LiDAR):

```bash
python tools/render.py outputs/<experiment_dir>            # everything
python tools/render.py outputs/<experiment_dir> --stride 5 # every 5th frame
python tools/render.py outputs/<experiment_dir> --skip-lidar
```

## Planned features

General features: 

- [ ] **Native SMPL pose extraction** (coming soon, being tested). Script working directly on the generic dataset provider interface, so pedestrian reconstruction won't require externally preprocessed human poses
- [ ] **Fast rendering API on inference** (coming soon, being refactored) 
- [ ] **Real-time desktop demo** The one showed in header demo video. Currently local inference only. Requires heavy refactoring

Dataset support status: 
- [x] **[Waymo Open Dataset](https://waymo.com/open/)** Motion v2, parquet version supported
- [ ] **[Argoverse 2](https://www.argoverse.org/av2.html)** — coming soon, being refactored
- [ ] **[PandaSet](https://pandaset.org/)** — coming soon, being refactored
- [ ] **[nuPlan](https://www.nuscenes.org/nuplan)** — TODO
- [ ] **[nuScenes](https://www.nuscenes.org/nuscenes)** — TODO

New datasets plug in through the common provider API (`xsim.data.provider.DatasetScene`) -- generic scene/data log description.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{patakin2026xsim,
  title     = {Unified Sensor Simulation for Autonomous Driving},
  author    = {Patakin, Nikolay and Shirokov, Arsenii and Konushin, Anton and Senushkin, Dmitry},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2026}
}
```

## Acknowledgements

XSIM builds on ideas and code from several excellent open-source projects:

- [gsplat](https://github.com/nerfstudio-project/gsplat), [3DGUT](https://github.com/nv-tlabs/3dgrut)  — densification strategies
- [OmniRe / drivestudio](https://github.com/ziyc/drivestudio) — deformation and human-body modeling utilities
- [SplatAD](https://github.com/carlinds/splatad) / [neurad-studio](https://github.com/georghess/neurad-studio) — CNN post-processing
- [SMPL / SMPL-X](https://smpl-x.is.tue.mpg.de/) — human body model

## License

XSIM is released under the [Apache License 2.0](LICENSE).

Parts of the codebase contain modified code from third-party projects;
their licenses and copyright notices are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). 
The SMPL model **data** it loads is licensed separately by the Max Planck Institute 
for **non-commercial scientific research use only** and is not distributed with this repository.
