"""Fig. 8 (training-loss convergence, Transformer, from the logged console values) and
Fig. 9 (paired per-patient AUC, single-slice vs Transformer). Reads stats.json."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("stats.json")); B, O = "#1f77b4", "#e8710a"
loss = {1: [.3794, .1639, .1392, .1128, .0658], 2: [.3560, .1664, .1414, .1103, .1101], 3: [.3508, .1478, .1389, .1564, .1085],
        4: [.3935, .1988, .1290, .1244, .0952], 5: [.3738, .1604, .1539, .1038, .1104]}
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
ep = np.arange(1, 6)
for k, v in loss.items(): ax[0].plot(ep, v, marker="o", ms=3, lw=1, color="#999999", alpha=.8, label="Fold %d" % k if False else None)
m = np.mean(list(loss.values()), axis=0); ax[0].plot(ep, m, marker="o", color=O, lw=2.2, label="Mean of 5 folds")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training cross-entropy"); ax[0].set_xticks(ep); ax[0].legend(frameon=False)
ax[0].set_title("A  Training loss per fold (Transformer)", loc="left", fontsize=9.5, fontweight="bold")
fa = S["folds"]; x = np.arange(1, 6)
ax[1].plot(x, fa["single"], marker="s", color=B, label="Single slice"); ax[1].plot(x, fa["transformer"], marker="o", color=O, label="Transformer")
ax[1].axhline(.5, color="#bbbbbb", ls="--", lw=.8); ax[1].set_ylim(.4, 1); ax[1].set_xticks(x); ax[1].set_xlabel("Held-out fold"); ax[1].set_ylabel("Test AUC")
ax[1].legend(frameon=False, loc="lower right"); ax[1].set_title("B  Held-out AUC per fold", loc="left", fontsize=9.5, fontweight="bold")
plt.tight_layout(); 
for e in ("png", "tif"): plt.savefig("figures/Fig8_convergence." + e, dpi=300)
plt.close()
P = [p for p in S["per_patient"] if p["single"]["auc"] is not None and p["transformer"]["auc"] is not None]
a = np.array([p["single"]["auc"] for p in P]); b = np.array([p["transformer"]["auc"] for p in P]); d = b - a
o = np.argsort(d)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
for i in range(len(P)): ax[0].plot([0, 1], [a[i], b[i]], color=O if d[i] > 0 else B, alpha=.55, lw=1)
ax[0].plot([0, 1], [np.median(a), np.median(b)], color="k", lw=2.4, marker="o", label="Median")
ax[0].axhline(.5, color="#bbbbbb", ls="--", lw=.8); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Single slice", "Transformer"]); ax[0].set_xlim(-.2, 1.2)
ax[0].set_ylabel("Within-patient AUC"); ax[0].legend(frameon=False, loc="lower center")
ax[0].set_title("A  Paired patients", loc="left", fontsize=9.5, fontweight="bold")
ax[1].bar(range(len(P)), d[o], color=[O if v > 0 else B for v in d[o]]); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xticks(range(len(P))); ax[1].set_xticklabels([P[i]["patient"].replace("HCC_", "") for i in o], rotation=90, fontsize=6.5)
ax[1].set_ylabel("ΔAUC (Transformer − single)"); ax[1].set_xlabel("Patient ID (HCC_xxx), sorted")
ax[1].set_title("B  Per-patient change (%d of %d improved)" % ((d > 0).sum(), len(d)), loc="left", fontsize=9.5, fontweight="bold")
plt.tight_layout()
for e in ("png", "tif"): plt.savefig("figures/Fig9_paired_patients." + e, dpi=300)
print(len(P), (d > 0).sum(), (d < 0).sum(), np.median(d), np.median(a), np.median(b))
