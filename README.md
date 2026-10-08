# LiverSense: slice-level HCC tumour detection (HCC-TACE-Seg)

Scripts for the paper "Slice-level detection of hepatocellular carcinoma on contrast-enhanced CT: patient-level evaluation of CNN and slice-sequence models with leakage and shortcut controls".

No images or patient data are included. Download HCC-TACE-Seg from The Cancer Imaging Archive.

## Steps
1. `python inspect_seg.py`            - inventory of DICOM-SEG objects, links to CT slices
2. `python build_lesion_dataset.py`   - builds lesion_X.npy and lesion_meta.csv
3. `python train_lesion.py --baselines-only`
4. `python train_lesion.py --modes single mean transformer bilstm --epochs 5 --tag _all`

Edit the default folder paths at the top of each script (Windows paths are used as defaults).

Requirements: Python 3, numpy, pandas, scikit-learn, pydicom, opencv-python, torch, torchvision. Python 3.14.7, torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1, pydicom 3.0.2 (numpy, pandas and opencv-python: current releases)

## Licence
MIT
