"""Deterministically rebuild the external WMT14 / XSum train/val splits (same as data_ext/*.pickle).
usage: python build_ext.py <task> <official_data_dir> <out.pickle>"""
import sys, pickle, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
task, dd, out = sys.argv[1:4]
rng = np.random.default_rng(0)
if task == "wmt14":
    tr = pd.read_parquet(hf_hub_download("wmt/wmt14", "de-en/train-00000-of-00003.parquet", repo_type="dataset"))
    va = pd.read_parquet(hf_hub_download("wmt/wmt14", "de-en/validation-00000-of-00001.parquet", repo_type="dataset"))
    f = lambda df: [(t["de"], t["en"]) for t in df.translation]
    T = [x for x in f(tr) if 5 <= len(x[1].split()) <= 40]; T = [T[i] for i in rng.choice(len(T), 5000, replace=False)]
    V = f(va); V = [V[i] for i in rng.choice(len(V), 1000, replace=False)]
    test = set(pickle.load(open(f"{dd}/sources.pickle", "rb"))); assert not any(s in test for s, _ in T + V)
    D = {"train_src": [s for s, _ in T], "train_ref": [r for _, r in T], "val_src": [s for s, _ in V], "val_ref": [r for _, r in V]}
else:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("gpt2")
    # consume the WMT rng draws first so the XSum stream matches the local build (both used one rng)
    _tr = pd.read_parquet(hf_hub_download("wmt/wmt14", "de-en/train-00000-of-00003.parquet", repo_type="dataset"))
    _n = sum(1 for t in _tr.translation if 5 <= len(t["en"].split()) <= 40); rng.choice(_n, 5000, replace=False); del _tr
    _va = pd.read_parquet(hf_hub_download("wmt/wmt14", "de-en/validation-00000-of-00001.parquet", repo_type="dataset")); rng.choice(len(_va), 1000, replace=False)
    def build(df, n):
        o, r = [], []
        for i in rng.permutation(len(df)):
            d, s = df.document.iloc[i], df.summary.iloc[i]
            if len(tk(s)["input_ids"]) > 48 or len(d) < 50: continue
            o.append(tk(d)["input_ids"][:512]); r.append(s)
            if len(o) >= n: break
        return o, r
    a, b = build(pd.read_parquet(hf_hub_download("EdinburghNLP/xsum", "data/train-00000-of-00001.parquet", repo_type="dataset")), 5000)
    c, d = build(pd.read_parquet(hf_hub_download("EdinburghNLP/xsum", "data/validation-00000-of-00001.parquet", repo_type="dataset")), 1000)
    test = set(pickle.load(open(f"{dd}/reference_summaries.pickle", "rb")))
    ka = [i for i, x in enumerate(b) if x not in test]; kc = [i for i, x in enumerate(d) if x not in test]
    D = {"train_src": [a[i] for i in ka], "train_ref": [b[i] for i in ka], "val_src": [c[i] for i in kc], "val_ref": [d[i] for i in kc]}
pickle.dump(D, open(out, "wb")); print(task, {k: len(v) for k, v in D.items()})
