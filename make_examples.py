"""
make_examples.py - figure of example slices (tumour-present vs tumour-absent) from your dataset.
Run in your PhD Work folder:   python make_examples.py
Needs lesion_X.npy and lesion_meta.csv.  Output: Fig_examples.png  (send it to me).
Pairs: for 6 patients, one tumour-present slice (from large/medium/small tumours) next to a tumour-absent slice of the SAME patient.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = r"D:\Praveen_PhD Work\PhD Work"
X = np.load(os.path.join(D, "lesion_X.npy")); m = pd.read_csv(os.path.join(D, "lesion_meta.csv"))
rng = np.random.default_rng(3)
pos = m[m.label == 1]
targets = [(10000, 1e9, "large (≥10,000 px)"), (10000, 1e9, "large (≥10,000 px)"), (2000, 10000, "medium (2,000–9,999 px)"),
           (2000, 10000, "medium (2,000–9,999 px)"), (30, 500, "small (30–499 px)"), (500, 2000, "small–medium (500–1,999 px)")]
used, rows = set(), []
for lo, hi, name in targets:
    c = pos[(pos.tumour_px >= lo) & (pos.tumour_px < hi) & (~pos.patient.isin(used))]
    if len(c) == 0: continue
    r = c.iloc[rng.integers(len(c))]; used.add(r.patient)
    neg = m[(m.patient == r.patient) & (m.label == 0)]
    if len(neg) == 0: continue
    rows.append((r, neg.iloc[rng.integers(len(neg))], name))
fig, ax = plt.subplots(2, len(rows), figsize=(2.4 * len(rows), 5.2))
for j, (p, n, name) in enumerate(rows):
    for i, (r, lab) in enumerate([(p, "tumour present"), (n, "tumour absent")]):
        a = ax[i, j]; a.imshow(X[r.name], cmap="gray", vmin=0, vmax=255); a.axis("off")
        a.set_title(f"{lab}\n{r.patient}, z={r.z_mm:.0f} mm" + (f"\n{name}" if i == 0 else "\n(same patient)"), fontsize=7)
plt.tight_layout(h_pad=3.5); plt.savefig(os.path.join(D, "Fig_examples.png"), dpi=300, facecolor="white")
print("saved Fig_examples.png")
