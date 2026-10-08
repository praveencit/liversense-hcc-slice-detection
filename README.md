# LiverSense: slice-level HCC tumour detection (HCC-TACE-Seg)

Scripts for the paper "Slice-level detection of hepatocellular carcinoma on contrast-enhanced CT: patient-level evaluation of CNN and slice-sequence models with leakage and shortcut controls".

No images or patient data are included. Download HCC-TACE-Seg (CT and SEG series) from The Cancer Imaging Archive: https://doi.org/10.7937/tcia.5fna-0924

## Steps
1. `python inspect_seg.py`            - inventory of DICOM-SEG objects and their links to CT slices (seg_inventory.csv)
2. `python build_lesion_dataset.py`   - builds lesion_X.npy and lesion_meta.csv
3. `python train_lesion.py --baselines-only`  - logistic-regression baselines, with and without leakage
4. `python train_lesion.py --modes single --epochs 5 --tag _single` (also: transformer, mean, bilstm)
5. `python analyze_results.py --data-dir <folder> --out-dir figures` - statistics, tables and figures of the paper
6. `python make_examples.py`          - optional figure of example slices

Edit the default folder paths at the top of each script (Windows paths are used as defaults).

## Requirements
Python 3.14.7, torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1, pydicom 3.0.2, numpy, pandas, opencv-python, matplotlib, pillow (current releases). `pip install numpy pandas scikit-learn pydicom opencv-python matplotlib torch torchvision`

## Licence
MIT
