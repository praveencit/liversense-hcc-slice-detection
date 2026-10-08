"""
analyze_results.py - recompute all statistics, tables and figures of the paper from the saved outputs.
Inputs (in --data-dir): lesion_meta.csv and lesion_oof_*.npz (written by train_lesion.py).
Outputs (in --out-dir): stats.json, Fig2_architecture, Fig3_cohort, Fig4_results, Fig5_diagnostics, Fig6_folds_size (.png/.tif).
Usage:  python analyze_results.py --data-dir "D:\\Praveen_PhD Work\\PhD Work" --out-dir figures
"""
import argparse, glob, json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=r"D:\Praveen_PhD Work\PhD Work"); ap.add_argument("--out-dir", default="figures"); ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True)
m = pd.read_csv(os.path.join(A.data_dir, "lesion_meta.csv")); y = m.label.values; pat = m.patient.values
ups = np.array(sorted(set(pat)))
folds = []
for tr, te in KFold(5, shuffle=True, random_state=A.seed).split(ups):
    folds.append(np.where(np.isin(pat, ups[te]))[0])
fold_of = np.zeros(len(y), int)
for i_, te_ in enumerate(folds): fold_of[te_] = i_ + 1
runs = {}
for fn in sorted(glob.glob(os.path.join(A.data_dir, "lesion_oof_*.npz"))):
    if "quick" in fn: continue
    z = np.load(fn, allow_pickle=True); assert (z["y"] == y).all()
    for key in z.files:
        if key.endswith(f"_seed{A.seed}"): runs[key.rsplit("_seed", 1)[0]] = z[key]
idx_by={u:np.where(pat==u)[0] for u in ups}
def boot(fn,n=2000,seed=0):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        pick=rng.choice(ups,len(ups),replace=True); ix=np.concatenate([idx_by[u] for u in pick])
        if len(set(y[ix]))<2: continue
        v.append(fn(ix))
    v=np.array(v); return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
S={"folds":{}, "models":{}, "diff":{}, "patients_per_fold":[len(set(pat[te])) for te in folds]}
for k,p in runs.items():
    d={}
    d["auc"]=roc_auc_score(y,p); d["auc_ci"]=boot(lambda ix:roc_auc_score(y[ix],p[ix]))
    d["ap"]=average_precision_score(y,p); d["ap_ci"]=boot(lambda ix:average_precision_score(y[ix],p[ix]))
    d["brier"]=brier_score_loss(y,p)
    pr=p>=0.5; tp=int((pr&(y==1)).sum()); fn=int((~pr&(y==1)).sum()); tn=int((~pr&(y==0)).sum()); fp=int((pr&(y==0)).sum())
    d.update(tp=tp,fn=fn,tn=tn,fp=fp,sens=tp/(tp+fn),spec=tn/(tn+fp),ppv=tp/max(tp+fp,1),npv=tn/max(tn+fn,1),acc=(tp+tn)/len(y))
    d["sens_ci"]=boot(lambda ix:((p[ix]>=.5)&(y[ix]==1)).sum()/max((y[ix]==1).sum(),1))
    d["spec_ci"]=boot(lambda ix:((p[ix]<.5)&(y[ix]==0)).sum()/max((y[ix]==0).sum(),1))
    S["folds"][k]=[roc_auc_score(y[te],p[te]) for te in folds]
    pp=[roc_auc_score(y[idx_by[u]],p[idx_by[u]]) for u in ups]; d["pp_median"]=float(np.median(pp)); d["pp_iqr"]=[float(np.percentile(pp,25)),float(np.percentile(pp,75))]; d["pp_below_half"]=int(sum(a<0.5 for a in pp))
    pos=m.label==1; bins=[0,500,2000,5000,10000,1e9]; cat=pd.cut(m.tumour_px[pos],bins,labels=["30-499","500-1999","2000-4999","5000-9999",">=10000"])
    d["size_sens"]={str(c):[int((cat==c).sum()), float((p[pos.values][(cat==c).values]>=.5).mean())] for c in cat.cat.categories}
    # thickness / kVp strata (AUC)
    d["kvp_auc"]={str(v):float(roc_auc_score(y[m.kvp.values==v],p[m.kvp.values==v])) for v in sorted(m.kvp.unique())}
    d["thk_auc"]={"2.5mm":float(roc_auc_score(y[m.slice_thickness.values==2.5],p[m.slice_thickness.values==2.5])),"other":float(roc_auc_score(y[m.slice_thickness.values!=2.5],p[m.slice_thickness.values!=2.5]))}
    S["models"][k]=d
ks=list(runs)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b=ks[i],ks[j]; f=lambda ix:roc_auc_score(y[ix],runs[b][ix])-roc_auc_score(y[ix],runs[a][ix])
        S["diff"][f"{b}-{a}"]={"delta":float(roc_auc_score(y,runs[b])-roc_auc_score(y,runs[a])),"ci":boot(f)}
S["cohort"]={"n_pat":len(ups),"n":len(y),"pos":int(y.sum()),"fold_pats":[sorted(set(pat[te])) for te in folds],
  "fold_n":[int(len(te)) for te in folds],"fold_pos":[int(y[te].sum()) for te in folds]}

# ---- extended analyses (position, operating points, per-fold confusion, per-patient, ICC)
rel=m.groupby(["patient","series_tail"]).z_index.transform(lambda s:(s-s.min())/max(s.max()-s.min(),1)).values
edges=[0,0.2,0.4,0.6,0.8,1.0001]; S["position_labels"]=["0-20%","20-40%","40-60%","60-80%","80-100%"]
bin_of=np.digitize(rel,edges[1:-1])
S["position_n"]=[int((bin_of==b).sum()) for b in range(5)]; S["position_prev"]=[float(y[bin_of==b].mean()) for b in range(5)]
S["position"]={}
for k,p in runs.items():
    S["position"][k]={"sens":[float((p[(bin_of==b)&(y==1)]>=.5).mean()) for b in range(5)],"spec":[float((p[(bin_of==b)&(y==0)]<.5).mean()) for b in range(5)],
                       "auc":[float(roc_auc_score(y[bin_of==b],p[bin_of==b])) for b in range(5)]}
def sens_at_spec(ix,p,sp=0.90):
    fpr,tpr,_=roc_curve(y[ix],p[ix]); return float(tpr[fpr<=1-sp].max())
def spec_at_sens(ix,p,se=0.90):
    fpr,tpr,_=roc_curve(y[ix],p[ix]); return float(1-fpr[tpr>=se].min())
S["op"]={}
allix=np.arange(len(y))
for k,p in runs.items():
    S["op"][k]={"sens_at_90spec":sens_at_spec(allix,p),"sens_at_90spec_ci":boot(lambda ix:sens_at_spec(ix,p),n=1000),
                "spec_at_90sens":spec_at_sens(allix,p),"spec_at_90sens_ci":boot(lambda ix:spec_at_sens(ix,p),n=1000)}
S["fold_conf"]={}
for k,p in runs.items():
    r=[]
    for te in folds:
        pr=p[te]>=.5; yy=y[te]; r.append([int((pr&(yy==1)).sum()),int((pr&(yy==0)).sum()),int((~pr&(yy==1)).sum()),int((~pr&(yy==0)).sum())])
    S["fold_conf"][k]=r
pp=[]
for u in ups:
    ix=idx_by[u]; row={"patient":u,"n":int(len(ix)),"pos":int(y[ix].sum()),"kvp":int(m.kvp.values[ix][0]),"thk":float(m.slice_thickness.values[ix][0]),"fold":int(fold_of[ix][0])}
    for k,p in runs.items():
        row[k]={"auc":float(roc_auc_score(y[ix],p[ix])),"sens":float((p[ix][y[ix]==1]>=.5).mean()),"spec":float((p[ix][y[ix]==0]<.5).mean())}
    pp.append(row)
S["per_patient"]=pp
# intraclass correlation of the label within patients (one-way ANOVA) and design effect
ns=np.array([len(idx_by[u]) for u in ups]); N=ns.sum(); k_=len(ups); means=np.array([y[idx_by[u]].mean() for u in ups]); gm=y.mean()
msb=(ns*(means-gm)**2).sum()/(k_-1); msw=sum(((y[idx_by[u]]-means[i])**2).sum() for i,u in enumerate(ups))/(N-k_)
m0=(N-(ns**2).sum()/N)/(k_-1); icc=(msb-msw)/(msb+(m0-1)*msw); deff=1+(ns.mean()-1)*icc
S["icc"]={"icc":float(icc),"design_effect":float(deff),"effective_n":float(N/deff),"mean_cluster":float(ns.mean())}

json.dump(S, open(os.path.join(A.out_dir, "stats.json"), "w"), indent=1, default=float)
for k,d in S["models"].items(): print(k, round(d["auc"],3),[round(x,3) for x in d["auc_ci"]],round(d["ap"],3),round(d["sens"],3),round(d["spec"],3),"brier",round(d["brier"],3))
print(S["folds"], S["diff"], S["cohort"]["fold_n"], S["cohort"]["fold_pos"])
print(S["models"]["single"]["kvp_auc"],S["models"]["single"]["thk_auc"])

NAMES={"single":"Single slice","mean":"3 slices, mean","transformer":"3 slices, Transformer","bilstm":"3 slices, Bi-LSTM"}
COL={"single":"#1B6CA8","mean":"#7A7A7A","transformer":"#D55E00","bilstm":"#2E8B57"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False})
def save(fig,name):
    fig.savefig(f"{A.out_dir}/{name}.png",dpi=300,facecolor="white",bbox_inches="tight")
    Image.open(f"{A.out_dir}/{name}.png").convert("RGB").save(f"{A.out_dir}/{name}.tif",dpi=(300,300),compression="tiff_lzw"); plt.close(fig)
# ---------- Fig 2 architecture
fig,ax=plt.subplots(figsize=(7.6,4.4)); ax.set_xlim(0,13); ax.set_ylim(0,7.4); ax.axis("off")
def box(x,y0,w,h,t,fc="#EEF2F7",ec="#33415C",fs=7.2,bold=False):
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle="round,pad=0.02,rounding_size=0.1",fc=fc,ec=ec,lw=1.0)); ax.text(x+w/2,y0+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",linespacing=1.3)
def ar(x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#33415C",lw=1.1))
for i,lab in enumerate(["slice z−1","slice z (target)","slice z+1"]):
    yy=5.3-i*1.75
    box(0.1,yy,2.3,1.2,f"{lab}\n128×128\n(1 ch → 3 ch)",fc="#FFF4DE" if i==1 else "#EEF2F7")
    box(3.0,yy,2.4,1.2,"ResNet-18\nshared weights\nImageNet init",fc="#E3EEF8"); ar(2.4,yy+0.6,3.0,yy+0.6)
    box(6.0,yy,2.2,1.2,"512 → 256\nReLU + dropout\n(embedding)"); ar(5.4,yy+0.6,6.0,yy+0.6); ar(8.2,yy+0.6,8.8,3.7)
box(8.8,2.4,2.5,2.6,"Sequence module\n(one of)\nmean\nTransformer\n(2 layers, 4 heads)\nBi-LSTM\n(128 per direction)",fc="#E3F0E6",ec="#2F6B3F")
box(11.7,3.1,1.25,1.2,"centre\nread-out\n→ linear\n→ 2",fc="#FFF4DE",ec="#8A5A00",fs=6.8); ar(11.3,3.7,11.7,3.7)
ax.text(0.1,0.7,"“Single” variant: only the centre slice passes through ResNet-18 → 256-d embedding → linear layer.",fontsize=7.2,style="italic")
save(fig,"Fig2_architecture")
# ---------- Fig 3 cohort
g=m.groupby("patient").agg(n=("label","size"),pos=("label","sum")).sort_values("n",ascending=False)
fig,ax=plt.subplots(1,3,figsize=(9.8,3.3),gridspec_kw={"width_ratios":[1.7,1,1.15]})
a=ax[0]; x=np.arange(len(g)); a.bar(x,g.pos,color="#D55E00",label="tumour-present"); a.bar(x,g.n-g.pos,bottom=g.pos,color="#9DB7D1",label="tumour-absent")
a.set_xticks([]); a.set_xlabel("32 patients (sorted by slice count)"); a.set_ylabel("Slices"); a.legend(frameon=False,fontsize=7.5); a.set_title("A  Slices per patient",loc="left",fontsize=9,fontweight="bold")
b=ax[1]; b.hist(np.log10(m.tumour_px[m.label==1]),bins=22,color="#D55E00",alpha=.9); b.set_xlabel("Tumour area (pixels, log10)"); b.set_ylabel("Positive slices"); b.set_title("B  Tumour area",loc="left",fontsize=9,fontweight="bold")
c=ax[2]; fp=S["cohort"]["fold_pats"]; ups=sorted(set(m.patient)); M=np.zeros((5,len(ups)))
for i,pl in enumerate(fp):
    for u in pl: M[i,ups.index(u)]=1
c.imshow(M,aspect="auto",cmap=matplotlib.colors.ListedColormap(["#E6ECF3","#1B6CA8"]),interpolation="nearest"); c.set_yticks(range(5)); c.set_yticklabels([f"Fold {i+1}" for i in range(5)]); c.set_xticks([]); c.set_xlabel("Patients (each column one patient)")
c.set_title("C  Test patients per fold",loc="left",fontsize=9,fontweight="bold")
plt.tight_layout(); save(fig,"Fig3_cohort")
# ---------- Fig 5 diagnostics
fig,ax=plt.subplots(1,4,figsize=(11.6,3.2))
a=ax[0]
for k,p in runs.items():
    pr,rc,_=precision_recall_curve(y,p); a.plot(rc,pr,c=COL[k],lw=1.5,label=NAMES[k])
a.axhline(y.mean(),ls="--",c="#999",lw=.9); a.set_xlabel("Recall (sensitivity)"); a.set_ylabel("Precision"); a.set_ylim(0.3,1.02); a.legend(frameon=False,fontsize=7); a.set_title("A  Precision–recall",loc="left",fontsize=9,fontweight="bold")
b=ax[1]; b.plot([0,1],[0,1],ls="--",c="#999",lw=.9)
for k,p in runs.items():
    fr,mp=calibration_curve(y,p,n_bins=8,strategy="quantile"); b.plot(mp,fr,"o-",c=COL[k],lw=1.3,ms=3.5,label=NAMES[k])
b.set_xlabel("Predicted probability"); b.set_ylabel("Observed fraction positive"); b.set_title("B  Calibration",loc="left",fontsize=9,fontweight="bold")
c=ax[2]; th=np.linspace(0.02,0.98,97)
for k,p in runs.items():
    c.plot(th,[((p>=t)&(y==1)).sum()/(y==1).sum() for t in th],c=COL[k],lw=1.5)
    c.plot(th,[((p<t)&(y==0)).sum()/(y==0).sum() for t in th],c=COL[k],lw=1.5,ls="--")
c.axvline(.5,c="#999",lw=.8); c.set_xlabel("Decision threshold"); c.set_ylabel("Sensitivity (solid) / specificity (dashed)"); c.set_title("C  Threshold sweep",loc="left",fontsize=9,fontweight="bold")
d=ax[3]; k0=list(runs)[-1] if len(runs)>1 else "single"; mm=S["models"][k0]; cm=np.array([[mm["tn"],mm["fp"]],[mm["fn"],mm["tp"]]])
d.imshow(cm,cmap="Blues"); 
for i in range(2):
    for j in range(2): d.text(j,i,f"{cm[i,j]}\n({cm[i,j]/cm[i].sum()*100:.0f}%)",ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=9)
d.set_xticks([0,1]); d.set_xticklabels(["pred. absent","pred. present"]); d.set_yticks([0,1]); d.set_yticklabels(["true absent","true present"]); d.set_title(f"D  Confusion ({NAMES[k0]})",loc="left",fontsize=9,fontweight="bold")
plt.tight_layout(); save(fig,"Fig5_diagnostics")

# ---------- Fig 7 position within the liver
fig,ax=plt.subplots(1,2,figsize=(8.2,3.2)); a=ax[0]
a.bar(range(5),S["position_prev"],color="#D55E00",width=0.6); a.set_xticks(range(5)); a.set_xticklabels([f"{l}\n(n={n})" for l,n in zip(S["position_labels"],S["position_n"])],fontsize=7.5)
a.set_ylim(0,1); a.set_ylabel("Fraction of slices with tumour"); a.set_xlabel("Relative position within the liver (inferior → superior)"); a.set_title("A  Tumour prevalence by position",loc="left",fontsize=9,fontweight="bold")
b=ax[1]; w=0.8/max(len(runs),1)
for i,(k,p) in enumerate(runs.items()):
    b.plot(range(5),S["position"][k]["sens"],"o-",c=COL[k],lw=1.5,ms=4,label=NAMES[k]+" sens."); b.plot(range(5),S["position"][k]["spec"],"s--",c=COL[k],lw=1.2,ms=3.5,label=NAMES[k]+" spec.")
b.set_xticks(range(5)); b.set_xticklabels(S["position_labels"],fontsize=7.5); b.set_ylim(0,1); b.set_xlabel("Relative position within the liver"); b.set_ylabel("Sensitivity / specificity at 0.5"); b.legend(frameon=False,fontsize=6.5,ncol=1); b.set_title("B  Performance by position",loc="left",fontsize=9,fontweight="bold")
plt.tight_layout(); save(fig,"Fig7_position")

# ---------- Fig 6 per-fold + size
fig,ax=plt.subplots(1,2,figsize=(8.2,3.2)); a=ax[0]; w=0.8/len(runs)
for i,(k,v) in enumerate(S["folds"].items()): a.bar(np.arange(5)+i*w-0.4+w/2,v,w,color=COL[k],label=NAMES[k])
a.set_xticks(range(5)); a.set_xticklabels([f"Fold {i+1}\n({n} pts)" for i,n in enumerate(S["patients_per_fold"])],fontsize=7.5); a.set_ylim(0.5,1); a.set_ylabel("AUC (held-out fold)"); a.legend(frameon=False,fontsize=7); a.set_title("A  AUC by fold",loc="left",fontsize=9,fontweight="bold")
b=ax[1]; groups=[("kvp_auc","100","100 kVp\n(6 pts)"),("kvp_auc","120","120 kVp\n(26 pts)"),("thk_auc","2.5mm","2.5 mm\n(29 pts)"),("thk_auc","other","3/5 mm\n(3 pts)")]
for i,k in enumerate(runs): b.bar(np.arange(4)+i*w-0.4+w/2,[S["models"][k][g][key] for g,key,_ in groups],w,color=COL[k])
b.axhline(0.5,ls="--",c="#999",lw=.9); b.set_xticks(range(4)); b.set_xticklabels([g[2] for g in groups],fontsize=7.5); b.set_ylim(0.4,1); b.set_ylabel("AUC (pooled within subgroup)"); b.set_title("B  Acquisition subgroups",loc="left",fontsize=9,fontweight="bold")
plt.tight_layout(); save(fig,"Fig6_folds_size")
print("figs ok")

fig,ax=plt.subplots(1,3,figsize=(9.6,3.4),gridspec_kw={"width_ratios":[1,1.15,1]})
a=ax[0]; a.plot([0,1],[0,1],ls="--",c="#999",lw=0.9)
for k,p in runs.items():
    fpr,tpr,_=roc_curve(y,p); a.plot(fpr,tpr,c=COL[k],lw=1.6,label=f"{NAMES[k]} (AUC {roc_auc_score(y,p):.2f})")
a.set_xlabel("1 − specificity"); a.set_ylabel("Sensitivity"); a.legend(frameon=False,fontsize=7.5,loc="lower right"); a.set_title("A  Pooled out-of-fold ROC",loc="left",fontsize=9,fontweight="bold")
b=ax[1]; pos=m[m.label==1]; bins=[0,500,2000,5000,10000,1e9]; lab=["30–500","500–2k","2k–5k","5k–10k",">10k"]
cat=pd.cut(pos.tumour_px,bins,labels=lab); n=cat.value_counts().reindex(lab).values; w=0.8/max(len(runs),1)
for i,(k,p) in enumerate(runs.items()):
    hit=(p[m.label.values==1]>=0.5); s=[hit[(cat==l).values].mean() for l in lab]
    b.bar(np.arange(5)+i*w-0.4+w/2,s,w,color=COL[k],label=NAMES[k])
b.set_xticks(range(5)); b.set_xticklabels([f"{l}\n(n={c})" for l,c in zip(lab,n)],fontsize=7.5)
b.set_xlabel("Tumour area on slice (pixels at 512×512)"); b.set_ylabel("Sensitivity at threshold 0.5"); b.set_ylim(0,1); b.set_title("B  Sensitivity by tumour size",loc="left",fontsize=9,fontweight="bold")
c=ax[2]; ks=list(runs)[:1] if len(runs)<2 else ["single",[k for k in runs if k!="single"][-1]]
ups=sorted(set(pat))
for j,k in enumerate(ks):
    v=[roc_auc_score(y[pat==u],runs[k][pat==u]) for u in ups]
    c.scatter(np.full(len(v),j)+np.random.default_rng(1).uniform(-0.12,0.12,len(v)),v,s=14,c=COL[k],alpha=0.85)
    c.hlines(np.median(v),j-0.25,j+0.25,color="k",lw=1.4)
c.axhline(0.5,ls="--",c="#999",lw=0.9); c.set_xticks(range(len(ks))); c.set_xticklabels([NAMES[k] for k in ks]); c.set_xlim(-0.6,len(ks)-0.4)
c.set_ylabel("Per-patient AUC (32 patients)"); c.set_ylim(0,1.02); c.set_title("C  Per-patient AUC (bar = median)",loc="left",fontsize=9,fontweight="bold")
if len(runs)>1: b.legend(frameon=False,fontsize=7,loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(A.out_dir,"Fig4_results.png"),dpi=300,facecolor="white")
Image.open(os.path.join(A.out_dir,"Fig4_results.png")).convert("RGB").save(os.path.join(A.out_dir,"Fig4_results.tif"),dpi=(300,300),compression="tiff_lzw")
