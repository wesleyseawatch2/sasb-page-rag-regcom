# Task 1 BGE-M3 GPU Runbook

This document is the hand-off checklist for rerunning the local Task 1
retrieval ablation on a machine with an NVIDIA GPU (for example, an RTX 5070).
It is intentionally self-contained so that another operator or Claude Code can
continue the experiment without guessing the data contract, command-line
options, or output locations.

## What this experiment does

`src/task1/local_bge_ablation.py` performs a no-API, local A/B experiment:

1. Reuses the reconstructed 490-query diagnostic input.
2. Recomputes the existing word TF-IDF, character TF-IDF, and metric/unit-rule
   lanes from page text.
3. Adds multilingual BGE-M3 page/query embeddings.
4. Combines the lanes with weighted reciprocal-rank fusion.
5. Reranks the fused Top-50 text candidates with
   `BAAI/bge-reranker-v2-m3`.
6. Writes comparable retrieval metrics and per-query rankings.

The result is a **diagnostic retrieval study**, not an official Task 1 score:
the official query-level input, PDF mapping, gold-page contract, and evaluator
have not been independently confirmed. Do not replace the paper's official
Subtask 2 numbers with this experiment.

## Files needed on the new computer

The current public snapshot includes the participant JSON/CSV inputs, source
PDFs, and selected generated runs listed below. If you clone an older commit or
if the release permission changes, copy the files from a private location
instead. Do not commit model weights or API keys.

```text
runs/task1-dense-optimized/retrieval.jsonl
runs/task1-json-source-v2/pages.jsonl
```

The corresponding `data/test/`, `data/datasets/all/`, and `Training Set/PDF/`
inputs are also included in the authorized snapshot. If they are unavailable,
obtain the organizer-provided files and follow `src/task1/README.md` to run
`prepare`, `retrieve`, and the optimized dense configuration.

## GPU setup

From the repository root in PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch wheel selected by the official PyTorch
installer for the installed NVIDIA driver and Python version. Then verify the
GPU before downloading a large model:

```powershell
nvidia-smi
py -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

If `CUDA: False`, stop and fix the NVIDIA driver/PyTorch installation first;
the experiment must not silently fall back to CPU.

The RTX 5070 reference specification lists 12 GB GDDR7 and CUDA capability
12.0, which is sufficient for this BGE embedding/reranking configuration when
using one process and moderate batches. See the official specification:
<https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/>.

## Recommended RTX 5070 command

Use **one worker** for one GPU. Four workers would load four copies of each
model and can exhaust VRAM. Batching, rather than process multiplication, is
the efficient GPU parallelism here.

```powershell
py src/task1/local_bge_ablation.py `
  --device cuda `
  --workers 1 `
  --torch-threads 8 `
  --page-char-limit 1500 `
  --bge-max-length 512 `
  --reranker-max-length 512 `
  --embedding-batch-size 32 `
  --reranker-batch-size 16 `
  --candidate-k 50 `
  --retrieval runs/task1-dense-optimized/retrieval.jsonl `
  --pages runs/task1-json-source-v2/pages.jsonl `
  --output runs/task1-bge-ablation/results.jsonl `
  --metrics-output artifacts/metrics/task1_bge_ablation.json
```

The first run downloads `BAAI/bge-m3` and
`BAAI/bge-reranker-v2-m3` from Hugging Face. The script caches page embeddings
under `cache/task1-bge-m3/`; a rerun with the same model, page limit, and
maximum length reuses those vectors. The model cards document BGE-M3's
multilingual embedding use and the reranker’s query/passage scoring:
<https://huggingface.co/BAAI/bge-m3> and
<https://huggingface.co/BAAI/bge-reranker-v2-m3>.

No `OPENAI_API_KEY` is needed and this run makes no paid API calls.

## Cheap smoke test before the full run

Run this first to check CUDA, model download, and VRAM. It embeds only the
first two reconstructed queries and skips the reranker:

```powershell
py src/task1/local_bge_ablation.py `
  --device cuda --workers 1 --torch-threads 8 `
  --page-char-limit 1500 --bge-max-length 512 `
  --embedding-batch-size 16 --max-queries 2 --skip-reranker `
  --output runs/task1-bge-gpu-smoke/results.jsonl `
  --metrics-output artifacts/metrics/task1_bge_gpu_smoke.json
```

While it runs, watch `nvidia-smi`. If VRAM is insufficient, reduce the
embedding batch to 8 and reranker batch to 4; do not increase worker count.

## Outputs to preserve

After a successful full run, preserve or commit only the small metrics/config
artifact (not raw PDFs or caches):

```text
artifacts/metrics/task1_bge_ablation.json
runs/task1-bge-ablation/results.jsonl       # keep private or archive separately
```

The metrics JSON reports `baseline_miniLM_existing`, `bge_fused`, and
`bge_reranked`, each with Hit@1/5/10, Near@1, MRR, and the number of evaluated
non-empty-gold rows. Record the exact command, Git commit, model revisions,
device, batch sizes, elapsed time, and GPU name in `docs/EXPERIMENT_LOG.md`.

## Interpretation checklist

- Compare BGE against the existing MiniLM/TF-IDF/rules baseline on exactly the
  same non-empty reconstructed gold rows.
- Report candidate recall before reranking and ranking metrics after reranking.
- Break down results by language and inspect at least 50 error cases if the
  method is used in the paper.
- Treat empty-gold rows separately; they are excluded from Hit/MRR averages.
- Do not tune weights or thresholds on the test-derived diagnostic labels and
  then call the resulting number official.
- If the run is interrupted, rerun the same command: completed page vectors are
  cached and the output/metrics files are replaced deterministically.

## Handoff definition of done

The GPU handoff is complete when `torch.cuda.is_available()` is true, the smoke
test completes without OOM, the full JSON metrics file exists, and the command
and environment are recorded. Only then update the paper with the result, and
label it as a local BGE ablation/diagnostic unless the organizers provide the
official Task 1 evaluator and data contract.
