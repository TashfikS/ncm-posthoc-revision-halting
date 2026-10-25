# Post-hoc latent revision and adaptive halting on frozen Next-Concept Models

Code, results and paper source for *"Why Post-Hoc Latent Revision and Adaptive Halting Fail on Frozen Coarse-to-Fine Language Models: Theoretical Conditions and Empirical Diagnostics"* (under review at TMLR).

**Author:** S M Tasfik Us Salahin

All experiments run the **unmodified** official [HazyResearch/ncm](https://github.com/HazyResearch/ncm) code with the released checkpoints (`hazyresearch/ncm-checkpoints`) and datasets (`hazyresearch/ncm-datasets`) on a single 16 GB T4 GPU.

## Layout
| Path | Content |
|---|---|
| `src/apc_lib.py` | Harness: wraps `get_next_autoregressive_input` so the official `generate()` runs unchanged; latent corrector C_ω, halting, token freezing, corrector training |
| `src/apc_v4.py` | Re-executed official level loop: codebook projection, per-sequence margin halting with batch compaction (bit-exact vs. official) |
| `src/run_task.py` | Main experiment driver (official evaluate.py reproduction, halting sweeps, 5-seed corrector runs, sensitivity) |
| `src/dagger.py` | Closed-loop generator co-adaptation (teacher-forced control / DAgger / DAgger + C_ω) |
| `scripts/` | `analyze.py` → `summary.json`; `gen_tex.py` → every number/table in the paper; `make_figs.py`; `flux_gt.py`, `diag_v4.py` diagnostics |
| `results/` | `summary.json`, `flux_gt.json`, `diag_v4.json`, `dagger_summary.txt`, and `raw/<run>/results/<task>/{results.json,log.txt}` for every T4 run (per-item scores included) |
| `paper/` | TMLR LaTeX source (`main.tex`) |
| `kaggle/` | Kaggle kernel metadata templates |

## Setup
```bash
git clone https://github.com/HazyResearch/ncm.git
pip install -r requirements.txt
export NCM_ROOT=$PWD/ncm
python - <<'PY'
from huggingface_hub import snapshot_download as s
for t in ["sudoku/easy","sudoku/medium","sudoku/hard","wmt14","xsum"]:
    s("hazyresearch/ncm-checkpoints", allow_patterns=[t+"/*"], local_dir="ck")
    s("hazyresearch/ncm-datasets", repo_type="dataset", allow_patterns=[t+"/*"], local_dir="data")
PY
```

## Reproducing
```bash
# Sudoku (per difficulty): official reproduction, halting sweeps, 5-seed correctors, sensitivity (hard)
python src/run_task.py --task sudoku-hard --ck ck/sudoku/hard --dd data/sudoku/hard --out out/sudoku-hard \
  --n_train 500 --n_test 1500 --bs 128 --epochs 40 --official_n 0 --zs_full --sens
#   validation-selected variant: add --skip_sweeps ; decode-path variant: add --skip_sweeps --corr_mode decode
python src/run_task.py --task wmt14 --ck ck/wmt14 --dd data/wmt14 --out out/wmt14 --n_train 1000 --n_test 2003 --bs 64 --epochs 20 --official_n 0
python src/run_task.py --task xsum  --ck ck/xsum  --dd data/xsum  --out out/xsum  --n_train 500 --n_test 1000 --bs 32 --epochs 30 --official_n 1500
# Generator co-adaptation
python src/dagger.py --task sudoku-hard --ck ck/sudoku/hard --dd data/sudoku/hard --out out/dagger-hard \
  --n_train 500 --n_test 1500 --rounds 3 --epochs 4 --bs 32 --ebs 250
# Tables, macros and figures from the released results
cd scripts && ln -s ../results/raw kout && python analyze.py && python gen_tex.py && python make_figs.py
```
Splits: the public repository contains only test files, so the first `n_train` items of each official test file are used for training/selection and the next `n_test` for evaluation. Seeds: 42, 0, 1, 2, 3.

## Data availability
No new data are created. Checkpoints and datasets are the public releases `hazyresearch/ncm-checkpoints` and `hazyresearch/ncm-datasets` on the Hugging Face Hub. Our per-run outputs (including per-item scores) are in `results/`.

## License
MIT (this code). The NCM code, checkpoints and datasets are subject to their own licenses.

## Citation
```bibtex
@misc{salahin2026posthoc,
  title  = {Why Post-Hoc Latent Revision and Adaptive Halting Fail on Frozen Coarse-to-Fine Language Models: Theoretical Conditions and Empirical Diagnostics},
  author = {Salahin, S M Tasfik Us},
  year   = {2026},
  note   = {Under review at Transactions on Machine Learning Research}
}
```
