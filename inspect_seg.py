"""
inspect_seg.py  -  header-only inspection of the DICOM-SEG objects in HCC-TACE-Seg.
Answers: which segments exist (liver? tumour? vessels?), how many slices each covers,
and which CT series each mask belongs to (so slices can be matched to masks).
Run from your PhD Work folder:   python inspect_seg.py
Send me seg_inventory.csv and the printed summary.
"""
import argparse, csv, os, collections
import pydicom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"D:\Praveen_PhD Work\PhD Work\hcc_tace_seg")
    ap.add_argument("--out", default=r"D:\Praveen_PhD Work\PhD Work")
    a = ap.parse_args()
    rows, labels, nseg = [], collections.Counter(), 0
    link_ok = link_bad = 0
    for p in sorted(f for f in os.listdir(a.base) if f.startswith("HCC_")):
        ct_sop, ct_series, segs = {}, collections.Counter(), []
        for root, _d, fs in os.walk(os.path.join(a.base, p)):
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
                    ct_sop[str(h.SOPInstanceUID)] = str(h.SeriesInstanceUID)
                    ct_series[str(h.SeriesInstanceUID)] += 1
                elif m == "SEG":
                    segs.append((fp, h))
        for fp, h in segs:
            nseg += 1
            seg_labels = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", "")) for s in getattr(h, "SegmentSequence", [])}
            for l in seg_labels.values():
                labels[l] += 1
            ref_series = ""
            try:
                ref_series = str(h.ReferencedSeriesSequence[0].SeriesInstanceUID)
            except Exception:
                pass
            per = getattr(h, "PerFrameFunctionalGroupsSequence", [])
            frames_by_seg = collections.Counter()
            src = set()
            for fr in per:
                try:
                    frames_by_seg[int(fr.SegmentIdentificationSequence[0].ReferencedSegmentNumber)] += 1
                except Exception:
                    pass
                try:
                    src.add(str(fr.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID))
                except Exception:
                    pass
            matched = sum(1 for s in src if s in ct_sop)
            series_hit = {ct_sop[s] for s in src if s in ct_sop}
            ok = bool(src) and matched == len(src)
            link_ok += ok
            link_bad += (not ok)
            rows.append([p, os.path.basename(fp), getattr(h, "NumberOfFrames", ""), len(src), matched, len(series_hit),
                         ref_series[-8:], ";".join(f"{k}:{v}" for k, v in sorted(seg_labels.items())),
                         ";".join(f"{k}:{v}" for k, v in sorted(frames_by_seg.items())), getattr(h, "Rows", ""), getattr(h, "Columns", "")])
    path = os.path.join(a.out, "seg_inventory.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "file", "n_frames", "n_source_slices", "n_matched_to_local_CT", "n_ct_series_hit", "ref_series_tail",
                    "segment_labels", "frames_per_segment", "rows", "cols"])
        w.writerows(rows)
    print(f"SEG objects: {nseg}")
    print("Segment labels (count of SEG objects containing each):")
    for k, v in labels.most_common():
        print(f"   {k!r}: {v}")
    print(f"SEG objects whose every source slice is found in the local CT: {link_ok}; not fully matched: {link_bad}")
    print("Saved", path)


if __name__ == "__main__":
    main()
