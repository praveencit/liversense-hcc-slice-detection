# LiverSense: slice-level HCC tumour detection (HCC-TACE-Seg)

Scripts for the paper "Slice-level detection of hepatocellular carcinoma on contrast-enhanced CT: patient-level evaluation of CNN and slice-sequence models with leakage and shortcut controls".

No images or patient data are included. Download HCC-TACE-Seg from The Cancer Imaging Archive.

## Steps
1. `python inspect_seg.py`            - inventory of DICOM-SEG objects, links to CT slices
2. `python build_lesion_dataset.py`   - builds lesion_X.npy and lesion_meta.csv
3. `python train_lesion.py --baselines-only`
4. `python train_lesion.py --modes single mean transformer bilstm --epochs 5 --tag _all`

Edit the default folder paths at the top of each script (Windows paths are used as defaults).

Requirements: Python 3, numpy, pandas, scikit-learn, pydicom, opencv-python, torch, torchvision. Python 3.14.7,
Name: torch
Version: 2.14.1
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org
Author:
Author-email: PyTorch Team <packages@pytorch.org>
License-Expression: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT
Location: C:\Users\Dell\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages
Requires: filelock, fsspec, jinja2, networkx, setuptools, sympy, typing-extensions
Required-by: torchvision
---
Name: torchvision
Version: 0.29.1
Summary: image and video datasets and models for torch deep learning
Home-page: https://github.com/pytorch/vision
Author: PyTorch Core Team
Author-email: soumith@pytorch.org
License: BSD
Location: C:\Users\Dell\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages
Requires: numpy, pillow, torch
Required-by:
---
Name: scikit-learn
Version: 1.9.1
Summary: A set of python modules for machine learning and data mining
Home-page: https://scikit-learn.org
Author:
Author-email:
License-Expression: BSD-3-Clause
Location: C:\Users\Dell\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages
Requires: joblib, narwhals, numpy, scipy, threadpoolctl
Required-by:
---
Name: pydicom
Version: 3.0.2
Summary: A pure Python package for reading and writing DICOM data
Home-page: https://github.com/pydicom/pydicom
Author:
Author-email: Darcy Mason and contributors <darcymason@gmail.com>
License:
Location: C:\Users\Dell\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages
Requires:
Required-by:

## Licence
MIT
