# To Meme or To Not Meme

This repository contains experiments for meme template identification in both supervised and unsupervised settings.

The original `meme-research/` folder preserves older research code and notebook-era implementations. The main runnable entry points for current experiments are in [SEED](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED).

## Repo Structure

- [SEED](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED): standalone evaluation scripts for fixed-split experiments and remote runs
- [meme-research](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/meme-research): original research code used as reference for method recreation
- [data](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/data): local data assets
- [notebooks](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/notebooks): exploratory notebooks
- [analysis](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/analysis) and [plots](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/plots): analysis outputs

## Main Idea

The current workflow is:

1. Create one canonical `80/20` ImgFlip train/test split.
2. Reuse that exact split across all methods.
3. Run one script per method.
4. Optionally run multiple seeds per method.
5. Collect metrics and timing from each batch summary.

This setup is intended to make comparisons fair across:

- `MLR`
- `rNN + pHash`
- `rNN + DenseNet embedding`
- `rNN + Feature Matching`
- `CNN` with `ResNet-18`, `DenseNet-121`, and `EfficientNetV2-S`
- `Two-stage DenseNet-121`
- `Sparse Matching`
- `SEED` unsupervised baseline
- `UMAP + HDBSCAN`
- `Zannettou pHash + DBSCAN`

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Some scripts download pretrained vision weights from `torchvision` or Hugging Face the first time they run. On remote machines, SSL certificate issues are common. If that happens, set the CA bundle explicitly:

```bash
python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
```

## Dataset Format

The evaluation scripts support two input styles:

1. Folder-of-folders dataset

Expected structure:

```text
/path/to/imgflip_root/
  template_a/
    img1.jpg
    img2.jpg
  template_b/
    img3.jpg
    img4.jpg
```

2. Parquet manifest

Supported columns:

- `image_path` and `template`
- or `path` and `template_name`

If the parquet paths need rewriting, use:

- `--image-root`
- `--path-prefix-from`
- `--path-prefix-to`

## Fixed Split Workflow

Create the split once:

```bash
python SEED/create_imgflip_split.py \
  --dataset-root "/Volumes/huysuy05/ssd_data/meme_match_data" \
  --output-dir SEED/splits/imgflip_80_20 \
  --train-size 0.8 \
  --random-seed 42
```

This writes:

- `SEED/splits/imgflip_80_20/train.parquet`
- `SEED/splits/imgflip_80_20/test.parquet`
- `SEED/splits/imgflip_80_20/split_config.json`

Once the split exists, every method should use the same two parquet files.

## Method Scripts

### Supervised

- [meme_research_mlr_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_mlr_eval.py)
- [meme_research_rnn_phash_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_rnn_phash_eval.py)
- [meme_research_rnn_densenet_embedding_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_rnn_densenet_embedding_eval.py)
- [meme_research_rnn_feature_matching_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_rnn_feature_matching_eval.py)
- [meme_research_cnn_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_cnn_eval.py)
- [meme_research_two_stage_cnn_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_two_stage_cnn_eval.py)
- [meme_research_sparse_matching_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/meme_research_sparse_matching_eval.py)

### Unsupervised

- [seed_unsupervised_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/seed_unsupervised_eval.py)
- [bertopic_clip_phash_hdbscan_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/bertopic_clip_phash_hdbscan_eval.py)
- [zannettou_phash_dbscan_eval.py](/Users/huy.suy05./Documents/Projects/To_Meme_or_To_Not_Meme/SEED/zannettou_phash_dbscan_eval.py)

## Example Commands

All commands below assume the fixed split already exists.

### MLR

```bash
python SEED/meme_research_mlr_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/mlr \
  --seeds 42,43,44
```

### rNN + pHash

```bash
python SEED/meme_research_rnn_phash_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/rnn_phash \
  --seeds 42,43,44
```

### rNN + DenseNet Embedding

```bash
python SEED/meme_research_rnn_densenet_embedding_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/rnn_densenet_embedding \
  --seeds 42,43,44
```

### rNN + Feature Matching

```bash
python SEED/meme_research_rnn_feature_matching_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/rnn_feature_matching \
  --max-refs-per-template 20 \
  --num-workers 8 \
  --seeds 42,43,44
```

### CNN

ResNet-18:

```bash
python SEED/meme_research_cnn_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --model-name resnet18 \
  --output-dir SEED/results/cnn_resnet18 \
  --seeds 42,43,44
```

DenseNet-121:

```bash
python SEED/meme_research_cnn_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --model-name densenet121 \
  --output-dir SEED/results/cnn_densenet121 \
  --seeds 42,43,44
```

EfficientNetV2-S:

```bash
python SEED/meme_research_cnn_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --model-name efficientnet_v2_s \
  --output-dir SEED/results/cnn_efficientnetv2 \
  --seeds 42,43,44
```

### Two-stage DenseNet-121

Only use this when you have negative non-meme images for stage 1:

```bash
python SEED/meme_research_two_stage_cnn_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --negative-root "/path/to/nonmemes" \
  --output-dir SEED/results/two_stage_densenet121 \
  --seeds 42,43,44
```

### Sparse Matching

```bash
python SEED/meme_research_sparse_matching_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/sparse_matching \
  --max-refs-per-template 30 \
  --num-workers 8 \
  --seeds 42,43,44
```

### SEED Unsupervised Baseline

```bash
python SEED/seed_unsupervised_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/seed_unsupervised_eval \
  --seeds 42,43,44
```

### UMAP + HDBSCAN

pHash:

```bash
python SEED/bertopic_clip_phash_hdbscan_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --embedding-method phash \
  --output-dir SEED/results/umap_hdbscan_phash \
  --seeds 42,43,44
```

CLIP:

```bash
python SEED/bertopic_clip_phash_hdbscan_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --embedding-method clip \
  --output-dir SEED/results/umap_hdbscan_clip \
  --seeds 42,43,44
```

### Zannettou pHash + DBSCAN

```bash
python SEED/zannettou_phash_dbscan_eval.py \
  --train-parquet SEED/splits/imgflip_80_20/train.parquet \
  --test-parquet SEED/splits/imgflip_80_20/test.parquet \
  --output-dir SEED/results/zannettou_phash_dbscan \
  --seeds 42,43,44
```

## Outputs

Each run creates a timestamped run directory under the requested `--output-dir`.

Single-seed runs write:

- `run_config.json`
- `test_metrics.json`
- `metrics_summary.csv`
- `test_predictions.csv`

Batch runs with `--seeds 42,43,44` additionally write:

- `batch_metrics_summary.csv`
- `batch_config.json`

Timing is recorded per seed and per batch:

- per seed:
  - `timing.duration_seconds`
  - `timing.duration_minutes`
- per batch:
  - `timing.average_duration_seconds`
  - `timing.average_duration_minutes`
  - `timing.total_duration_seconds`
  - `timing.total_duration_minutes`

## Metrics

Most methods report:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `mcc`
- `cohen_kappa`

Methods that can abstain or reject predictions also report:

- `coverage`
- `covered_accuracy`
- `covered_count`
- `rejected_count`

## Notes

- The fixed-split workflow is the recommended way to compare methods fairly.
- Multi-seed runs still matter even with a fixed outer split, because CNNs and clustering-based methods remain stochastic.
- `feature matching` and `sparse matching` are much slower than the other baselines.
- `Two-stage DenseNet-121` only matches the intended method when stage 1 uses a real negative dataset.
- `meme-research/` is kept as reference code. The main runnable scripts are in `SEED/`.
