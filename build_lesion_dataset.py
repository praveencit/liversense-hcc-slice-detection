"""
build_lesion_dataset.py  -  slice-level tumour-present / tumour-absent dataset from HCC-TACE-Seg.

For every patient whose DICOM-SEG object can be linked to the CT slices you have locally
(via ReferencedSOPInstanceUID), this script
  * reads the "Mass" (tumour) and "Liver" segments,
  * keeps only slices that contain liver (so negatives are liver slices, not chest/pelvis),
  * labels a slice 1 if the tumour mask has >= --min-px pixels, 0 if it has none,
    and DROPS ambiguous slices with 1..(min-px-1) tumour pixels,
  * converts the CT slice to Hounsfield units (RescaleSlope/Intercept) and applies a
    fixed abdominal window, resizes to --size and stores uint8.

Run from your PhD Work folder:
    python build_lesion_dataset.py
Outputs (in --out): lesion_X.npy (N,size,size) uint8, lesion_meta.csv, lesion_summary.txt
Send me lesion_summary.txt and lesion_meta.csv (not the .npy).
"""
import argparse, collections, csv, os
import numpy as np
import pydicom
import cv2

TUMOUR_LABELS = ("mass", "tumor", "tumour", "hcc", "lesion")


def find_files(base, patient):
    ct, seg = {}, []
    for root, _d, fs in os.walk(os.path.join(base, patient)):
        for f in fs:
            if not (f.endswith(".dcm") or "." not in f):
                continue
            fp = os.path.join(root, f)
            try:
                h = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:
                continue
            m = str(getattr(h, "Modality", ""))
            if m == "CT":
                ct[str(h.SOPInstanceUID)] = fp
            elif m == "SEG":
                seg.append(fp)
    return ct, seg


def to_hu(ds):
    a = ds.pixel_array.astype(np.float32)
    return a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"D:\Praveen_PhD Work\PhD Work\hcc_tace_seg")
    ap.add_argument("--out", default=r"D:\Praveen_PhD Work\PhD Work")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min-px", type=int, default=30, help="tumour pixels (at 512x512) for a positive slice")
    ap.add_argument("--min-liver-px", type=int, default=500)
    ap.add_argument("--win-lo", type=float, default=-140.0)
    ap.add_argument("--win-hi", type=float, default=260.0)
    a = ap.parse_args()

    X, meta, skipped = [], [], []
    for p in sorted(f for f in os.listdir(a.base) if f.startswith("HCC_")):
        ct, segs = find_files(a.base, p)
        for sp in segs:
            h = pydicom.dcmread(sp, force=True)
            labels = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", "")).lower() for s in h.SegmentSequence}
            tum = [n for n, l in labels.items() if l in TUMOUR_LABELS]
            liv = [n for n, l in labels.items() if l == "liver"]
            if not tum or not liv:
                skipped.append((p, "no tumour/liver segment")); continue
            arr = h.pixel_array
            if arr.ndim == 2:
                arr = arr[None]
            per = h.PerFrameFunctionalGroupsSequence
            tmask, lmask = {}, {}
            for i, fr in enumerate(per):
                try:
                    seg_no = int(fr.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                    sop = str(fr.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
                except Exception:
                    continue
                fm = arr[i] > 0
                if seg_no in tum:
                    tmask[sop] = tmask.get(sop, 0) | fm
                if seg_no in liv:
                    lmask[sop] = lmask.get(sop, 0) | fm
            sops = sorted(set(tmask) | set(lmask))
            if not sops or sum(s in ct for s in sops) < 0.9 * len(sops):
                skipped.append((p, f"CT not found locally ({sum(s in ct for s in sops)}/{len(sops)} slices)")); continue
            # z order from the CT headers
            zs = {}
            for s in sops:
                if s in ct:
                    d = pydicom.dcmread(ct[s], stop_before_pixels=True, force=True)
                    zs[s] = float(d.ImagePositionPatient[2])
            order = sorted(zs, key=lambda s: zs[s])
            for k, s in enumerate(order):
                lm, tm = lmask.get(s), tmask.get(s)
                lpx = int(lm.sum()) if lm is not None else 0
                tpx = int(tm.sum()) if tm is not None else 0
                if lpx < a.min_liver_px:
                    continue
                if 0 < tpx < a.min_px:
                    continue  # ambiguous
                ds = pydicom.dcmread(ct[s], force=True)
                if (ds.Rows, ds.Columns) != arr.shape[1:]:
                    skipped.append((p, "CT/SEG size mismatch")); break
                hu = to_hu(ds)
                img = np.clip((hu - a.win_lo) / (a.win_hi - a.win_lo), 0, 1)
                img = cv2.resize((img * 255).astype(np.uint8), (a.size, a.size), interpolation=cv2.INTER_AREA)
                X.append(img)
                meta.append([p, str(ds.SeriesInstanceUID)[-8:], s, k, f"{zs[s]:.2f}", lpx, tpx, int(tpx >= a.min_px),
                             getattr(ds, "SliceThickness", ""), getattr(ds, "Manufacturer", ""),
                             getattr(ds, "ManufacturerModelName", ""), getattr(ds, "KVP", "")])
            print(f"{p}: {sum(1 for m in meta if m[0] == p)} slices kept")

    X = np.stack(X) if X else np.zeros((0, a.size, a.size), np.uint8)
    np.save(os.path.join(a.out, "lesion_X.npy"), X)
    with open(os.path.join(a.out, "lesion_meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "series_tail", "sop", "z_index", "z_mm", "liver_px", "tumour_px", "label",
                    "slice_thickness", "manufacturer", "model", "kvp"])
        w.writerows(meta)

    pats = collections.defaultdict(lambda: [0, 0])
    for m in meta:
        pats[m[0]][m[7]] += 1
    lab = np.array([m[7] for m in meta])
    lines = [f"patients used: {len(pats)}", f"slices: {len(meta)}  tumour-present: {int(lab.sum())}  tumour-absent: {int((lab == 0).sum())}",
             f"patients with >=1 positive slice: {sum(1 for v in pats.values() if v[1] > 0)}",
             f"patients with >=1 negative slice: {sum(1 for v in pats.values() if v[0] > 0)}",
             f"skipped patients: {len(skipped)}"]
    lines += [f"   {p}: {why}" for p, why in skipped]
    lines.append("manufacturer/model counts (patients): " + str(dict(collections.Counter(
        {m[0]: (m[9], m[10]) for m in meta}.values()))))
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(a.out, "lesion_summary.txt"), "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
