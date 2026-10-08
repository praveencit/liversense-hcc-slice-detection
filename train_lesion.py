"""
train_lesion.py  -  tumour-present vs tumour-absent liver slices, patient-level cross-validation.

Prerequisite: lesion_X.npy and lesion_meta.csv from build_lesion_dataset.py (same folder).

    python train_lesion.py --baselines-only        # seconds, no deep learning
    python train_lesion.py --quick                 # smoke test: 2 folds, 1 epoch, ~minutes
    python train_lesion.py                         # full run: 5 folds x 4 models (hours on CPU)
    python train_lesion.py --modes single mean     # only some models

Models (all use an ImageNet-pretrained ResNet-18, 3-channel input as the network expects):
    single      : centre slice only                       (plain CNN baseline)
    mean        : ResNet embeddings of K neighbouring slices, averaged
    transformer : Transformer encoder over the K slice embeddings (learned positions), centre token read out
    bilstm      : Bi-LSTM over the K slice embeddings, centre step read out
The K slices are REAL consecutive slices of the same scan (z-1, z, z+1 ...), so the sequence
modules have an anatomically meaningful sequence to model.

Evaluation: GroupKFold by PATIENT (no patient appears in both train and test).
Outputs: lesion_results.csv, lesion_oof.npz, lesion_baselines.txt  (send me these).
"""
import argparse, csv, os, time
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DIR = r"D:\Praveen_PhD Work\PhD Work"


# ------------------------------------------------------------------ helpers
def patient_folds(patients, n_splits, seed):
    """list of (train_idx, test_idx) with whole patients per fold."""
    ups = np.array(sorted(set(patients)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pats = np.asarray(patients)
    out = []
    for tr, te in kf.split(ups):
        out.append((np.where(np.isin(pats, ups[tr]))[0], np.where(np.isin(pats, ups[te]))[0]))
    return out


def metrics(y, p, thr=0.5):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), acc=(tp + tn) / len(y))


def patient_bootstrap(y, p, patients, n=1000, seed=0):
    """95% CI of pooled AUC, resampling PATIENTS (slices of one patient are not independent)."""
    rng = np.random.default_rng(seed)
    ups = np.array(sorted(set(patients)))
    idx_by = {u: np.where(patients == u)[0] for u in ups}
    vals = []
    for _ in range(n):
        pick = rng.choice(ups, len(ups), replace=True)
        ix = np.concatenate([idx_by[u] for u in pick])
        if len(set(y[ix])) < 2:
            continue
        vals.append(roc_auc_score(y[ix], p[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def context_index(meta, K):
    """For each slice, indices of K neighbouring slices of the same scan (centre repeated if neighbour missing)."""
    key = {(r.patient, r.series_tail, r.z_index): i for i, r in enumerate(meta.itertuples())}
    half = K // 2
    ctx = np.zeros((len(meta), K), dtype=np.int64)
    for i, r in enumerate(meta.itertuples()):
        for j, d in enumerate(range(-half, half + 1)):
            ctx[i, j] = key.get((r.patient, r.series_tail, r.z_index + d), i)
    return ctx


# ------------------------------------------------------------------ baselines (no deep learning)
def baselines(X, meta, y, seed, out_dir):
    lines = []
    pat = meta.patient.values
    flat = X.reshape(len(X), -1).astype(np.float32) / 255.0
    hist = np.stack([np.histogram(f, bins=32, range=(0, 1))[0] / f.size for f in flat])
    F_int = np.column_stack([flat.mean(1), flat.std(1), hist])
    rel_z = meta.groupby(["patient", "series_tail"]).z_index.transform(lambda s: (s - s.min()) / max(s.max() - s.min(), 1))
    F_meta = np.column_stack([rel_z, meta.liver_px.values / 1e4])
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lines.append("Baselines, logistic regression, tumour-present vs absent (AUC; patient-level CV unless stated)")
    for name, F in [("intensity histogram", F_int), ("relative z-position + liver area", F_meta)]:
        oof = np.zeros(len(y))
        for tr, te in patient_folds(pat, 5, seed):
            oof[te] = clf().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        lo, hi = patient_bootstrap(y, oof, pat)
        lines.append(f"  {name:36s} patient-level split: AUC {roc_auc_score(y, oof):.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        oof2 = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
            oof2[te] = clf().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        lines.append(f"  {name:36s} random SLICE split (leaky): AUC {roc_auc_score(y, oof2):.3f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(out_dir, "lesion_baselines.txt"), "w", encoding="utf-8").write(txt)


# ------------------------------------------------------------------ deep models
def build_models():
    import torch, torch.nn as nn
    import torchvision

    class SeqNet(nn.Module):
        def __init__(self, mode, K, pretrained=True):
            super().__init__()
            self.mode, self.K = mode, K
            try:
                w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                r = torchvision.models.resnet18(weights=w)
            except Exception as e:  # offline: fall back to random init but say so
                print("WARNING: could not load ImageNet weights:", e)
                r = torchvision.models.resnet18(weights=None)
            r.fc = nn.Identity()
            self.cnn = r
            self.proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
            if mode == "transformer":
                self.pos = nn.Parameter(torch.zeros(1, K, 256))
                layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.1, batch_first=True)
                self.seq = nn.TransformerEncoder(layer, 2)
            elif mode == "bilstm":
                self.seq = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
            self.head = nn.Linear(256, 2)

        def forward(self, x):  # x: (B,K,H,W) in [0,1]
            B, K, H, W = x.shape
            if self.mode == "single":
                x = x[:, K // 2:K // 2 + 1]
                K = 1
            m = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            s = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            img = x.reshape(B * K, 1, H, W).repeat(1, 3, 1, 1)
            e = self.proj(self.cnn((img - m) / s)).view(B, K, 256)
            if self.mode == "single":
                z = e[:, 0]
            elif self.mode == "mean":
                z = e.mean(1)
            elif self.mode == "transformer":
                z = self.seq(e + self.pos)[:, K // 2]
            else:
                z = self.seq(e)[0][:, K // 2]  # (B,256) = 2 x 128 (forward+backward)
            return self.head(z)
    return SeqNet


def run_deep(X, meta, y, a, device_name):
    import torch, torch.nn as nn, torch.optim as optim
    SeqNet = build_models()
    device = torch.device(device_name)
    if a.size != X.shape[1]:
        X = np.stack([cv2.resize(x, (a.size, a.size), interpolation=cv2.INTER_AREA) for x in X])
    Xt = torch.tensor(X)  # uint8
    yt = torch.tensor(y, dtype=torch.long)
    pat = meta.patient.values
    ctx = context_index(meta, a.ctx)
    n_splits = 2 if a.quick else a.folds
    epochs = 1 if a.quick else a.epochs
    rows, store = [], {}

    def batch(idx, train):
        b = Xt[torch.as_tensor(ctx[idx])].float() / 255.0  # (B,K,H,W)
        if train:  # small random translation + brightness/contrast jitter
            dx, dy = np.random.randint(-8, 9, 2)
            b = torch.roll(b, shifts=(int(dy), int(dx)), dims=(2, 3))
            b = (b * (1 + 0.1 * (torch.rand(len(b), 1, 1, 1) - 0.5)) + 0.05 * (torch.rand(len(b), 1, 1, 1) - 0.5)).clamp(0, 1)
        return b.to(device), yt[idx].to(device)

    for mode in a.modes:
        for seed in a.seeds:
            print(f"\n>>> model={mode} seed={seed} device={device_name} epochs={epochs} size={a.size} K={a.ctx}")
            oof = np.zeros(len(y)); t0 = time.time()
            for f, (tr, te) in enumerate(patient_folds(pat, n_splits, seed)):
                torch.manual_seed(seed * 100 + f); np.random.seed(seed * 100 + f)
                model = SeqNet(mode, a.ctx, pretrained=not a.no_pretrained).to(device)
                opt = optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
                crit = nn.CrossEntropyLoss()
                for ep in range(epochs):
                    model.train(); perm = np.random.permutation(tr); tl = 0
                    for i in range(0, len(perm), a.batch_size):
                        bx, by = batch(perm[i:i + a.batch_size], True)
                        if len(bx) < 2:
                            continue
                        opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step(); tl += loss.item() * len(bx)
                    print(f"   fold {f + 1} epoch {ep + 1}/{epochs} train loss {tl / len(tr):.4f}")
                model.eval(); ps = []
                with torch.no_grad():
                    for i in range(0, len(te), 64):
                        bx, _ = batch(te[i:i + 64], False)
                        ps.append(torch.softmax(model(bx), 1)[:, 1].cpu().numpy())
                oof[te] = np.concatenate(ps)
                m = metrics(y[te], oof[te])
                print(f"   fold {f + 1}: AUC {m['auc']:.3f}  sens {m['sens']:.3f}  spec {m['spec']:.3f}  (test patients {len(set(pat[te]))})")
            ok = np.ones(len(y), bool) if not a.quick else np.isin(np.arange(len(y)), np.concatenate([te for _, te in patient_folds(pat, n_splits, seed)]))
            m = metrics(y[ok], oof[ok]); lo, hi = patient_bootstrap(y[ok], oof[ok], pat[ok])
            print(f"  pooled OOF [{mode} seed {seed}]: AUC {m['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f})  AP {m['ap']:.3f}  "
                  f"sens {m['sens']:.3f}  spec {m['spec']:.3f}  [{time.time() - t0:.0f}s]")
            rows.append([mode, seed, a.size, a.ctx, epochs, f"{m['auc']:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{m['ap']:.4f}",
                         f"{m['sens']:.4f}", f"{m['spec']:.4f}"])
            store[f"{mode}_seed{seed}"] = oof
    return rows, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--modes", nargs="+", default=["single", "mean", "transformer", "bilstm"],
                    choices=["single", "mean", "transformer", "bilstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=128, help="input resolution (256 is 4x slower)")
    ap.add_argument("--ctx", type=int, default=3, help="number of consecutive slices per sample (odd)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _single")
    ap.add_argument("--baselines-only", action="store_true")
    a = ap.parse_args()
    assert a.ctx % 2 == 1

    X = np.load(os.path.join(a.data_dir, "lesion_X.npy"))
    meta = pd.read_csv(os.path.join(a.data_dir, "lesion_meta.csv"), dtype={"series_tail": str})
    y = meta.label.values.astype(int)
    print(f"Loaded {X.shape}, patients {meta.patient.nunique()}, positive slices {y.sum()} / {len(y)}")
    baselines(X, meta, y, a.seeds[0], a.data_dir)
    if a.baselines_only:
        return
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, store = run_deep(X, meta, y, a, dev)
    suffix = ("_quick" if a.quick else "") + a.tag
    with open(os.path.join(a.data_dir, f"lesion_results{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "size", "ctx", "epochs", "AUC", "AUC_CI_lo", "AUC_CI_hi", "AP", "sensitivity", "specificity"])
        w.writerows(rows)
    np.savez(os.path.join(a.data_dir, f"lesion_oof{suffix}.npz"), y=y, patient=meta.patient.values, **store)
    print("\nSaved lesion_results.csv and lesion_oof.npz - send me these plus lesion_baselines.txt")


if __name__ == "__main__":
    main()
