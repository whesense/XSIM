# Third-party licenses

XSIM is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
Parts of the codebase contain modified code derived from the projects below.
The original copyright notices are reproduced here and, where applicable, in
the headers of the derived files. All derived code has been modified from its
original form.

## gsplat

- Upstream: https://github.com/nerfstudio-project/gsplat
- License: Apache License 2.0
- Copyright 2024-2025 the Regents of the University of California, Nerfstudio
  Team and contributors. All rights reserved.
- Used in: the Gaussian densification / pruning / relocation strategies in
  `src/xsim/modeling/gaussian/strategy/` (in particular the MCMC strategy).

## 3DGUT (3dgrut)

- Upstream: https://github.com/nv-tlabs/3dgrut
- License: Apache License 2.0
- Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- Used in: the Gaussian densification strategies in
  `src/xsim/modeling/gaussian/strategy/`.

## OmniRe (drivestudio)

- Upstream: https://github.com/ziyc/drivestudio
- License: MIT License
- Copyright (c) 2024 Ziyu Chen
- Used in: the deformation / human-body utilities in
  `src/xsim/modeling/nodes/gaussian/utils/` (deformation networks, voxel
  deformer, SMPL node utilities).

The full MIT license text of drivestudio:

```
MIT License

Copyright (c) 2024 Ziyu Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SplatAD / neurad-studio

- Upstream: https://github.com/carlinds/splatad and
  https://github.com/georghess/neurad-studio
- License: Apache License 2.0
- Copyright 2024 the authors of NeuRAD and contributors.
- Used in: the CNN post-processing node in
  `src/xsim/modeling/nodes/postprocess/cnn_postprocessor.py`.

## SMPL body model

The SMPL body model **data** (e.g. `SMPL_NEUTRAL`) licensed by the
Max Planck Institute for non-commercial scientific research, education, or
artistic projects. It is not distributed with this repository; to use the
human-body functionality you must register at https://smpl.is.tue.mpg.de/ and
obtain the model files under their license. If you use SMPL/SMPL-X, please
cite the papers below:

- Loper et al., "SMPL: A Skinned Multi-Person Linear Model", ACM Transactions
  on Graphics 34(6), SIGGRAPH Asia 2015.
- Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and Body from a
  Single Image", CVPR 2019 (SMPL-X).
