"""Embed src/*.py into a self-contained Kaggle script kernel: python scripts/build_kaggle_job.py <out_dir> "<run_task/dagger command args>" """
import base64, sys, os
out, args = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
emb = {n: base64.b64encode(open(f"src/{n}", "rb").read()).decode() for n in ("apc_lib.py", "apc_v4.py", "run_task.py", "dagger.py")}
open(f"{out}/run.py", "w").write(f'''import os, subprocess, base64
W = "/tmp/ncm_workspace"; os.makedirs(W, exist_ok=True)
def sh(c): print("$", c, flush=True); subprocess.run(c, shell=True)
sh("pip install -q omegaconf hydra-core einops sacrebleu rouge_score")
if not os.path.exists(W + "/ncm"): sh(f"git clone -q https://github.com/HazyResearch/ncm.git {{W}}/ncm")
for n, b in {emb!r}.items(): open(W + "/" + n, "wb").write(base64.b64decode(b))
sh(f"cd {{W}} && NCM_ROOT={{W}}/ncm python {args}")
''')
