# CodeBERT TTA Detection

This repository contains the experimental pipeline for evaluating test-time
adaptation (TTA) methods for AI-generated code detection. The source detector is
a GPTSniffer-style CodeBERT classifier fine-tuned on the Python split of HMCorp.
The project evaluates how this detector behaves under target-domain shifts and
whether lightweight TTA methods can improve robustness without target labels.

The codebase is adapted from the
[Blinorot PyTorch project template](https://github.com/Blinorot/pytorch_project_template/tree/main),
but the template training examples were removed in favor of an inference/TTA
pipeline.

## What Is Implemented

- GPTSniffer-style CodeBERT binary classifier loaded from HuggingFace.
- Unified `CodeDataset` for local JSONL files and HuggingFace datasets.
- Dynamic batch padding for tokenized code.
- Inference pipeline with Hydra configs.
- Metrics: accuracy, precision, recall, and F1.
- Optional experiment logging through WandB or Comet ML.
- TTA methods:
  - `none`: frozen source detector.
  - `tent`: TENT-style entropy minimization on LayerNorm affine parameters.
  - `tent` with accumulation: same method without resetting adapted weights after each batch.
  - `centroids`: SHOT-inspired centroid-based pseudo-label correction in embedding space.
  - `t2a`: lightweight T2A-inspired uncertainty-aware negative learning.
- Sanity-check baselines:
  - TF-IDF + logistic regression.
  - External CodeBERT detector evaluation.
  - Prediction-bias diagnostics for the source detector.

## Repository Layout

```text
.
├── inference.py                         # Hydra inference entry point
├── src/
│   ├── configs/                         # Hydra configs
│   │   ├── datasets/                    # Target datasets
│   │   ├── model/                       # GPTSniffer model config
│   │   ├── tta/                         # TTA method configs
│   │   └── writer/                      # WandB / Comet ML configs
│   ├── datasets/                        # Dataset and collate logic
│   ├── model/                           # GPTSniffer classifier wrapper
│   ├── metrics/                         # Classification metrics
│   ├── trainer/                         # Inferencer
│   └── tta/                             # TTA implementations
├── gptsniffer_finetuning/               # Fine-tuning utilities and local data
├── run_main.ipynb                       # Colab notebook for 4 datasets x 4 methods
├── run_t2a.ipynb                        # Colab notebook for T2A runs
├── run_aigcodeset_hyperparameters.ipynb # AIGCodeSet ablation runs
├── sklearn_logreg_baseline.py           # TF-IDF logistic regression baseline
├── codegendetect_codebert_eval.py       # External CodeBERT detector evaluation
├── gptsniffer_diagnostics.py            # Source detector bias diagnostics
└── extract*.py                          # WandB metric/plot extraction scripts
```

## Installation

Create an environment and install dependencies:

```bash
conda create -n coursework_1 python=3.10
conda activate coursework_1
pip install -r requirements.txt
```

If you use WandB logging:

```bash
wandb login
```

The default model config loads weights from:

```text
romangeek/hmcorp_python_gptsniffer
```

Set `HF_TOKEN` if HuggingFace rate limits become an issue.

## Data

The expected local HMCorp-derived files are:

```text
gptsniffer_finetuning/dataset/python/test.jsonl
gptsniffer_finetuning/dataset/python/test_no_comment.jsonl
gptsniffer_finetuning/dataset/java/test.jsonl
```

The fourth evaluation dataset is loaded from HuggingFace:

```text
basakdemirok/AIGCodeSet
```

Each dataset must contain a `code` field and one label field from:

```text
labels, label, generated, target
```

## Running Inference

The main entry point is:

```bash
python -u inference.py
```

By default, it uses:

- dataset: `gptsniffer_python_test`
- TTA: `none`
- writer: `wandb`
- batch size: `128`

Multi-line shell commands below use `\` as a line continuation character. They
can also be written as a single line by removing the backslashes and newlines.

### Disable Logging

For local smoke tests:

```bash
python -u inference.py \
  datasets=gptsniffer_python_test \
  tta=none \
  writer.mode=disabled \
  dataloader.batch_size=1 \
  +datasets.test.limit=1
```

### Run Without TTA

```bash
python -u inference.py \
  datasets=gptsniffer_python_test \
  tta=none \
  writer.mode=disabled
```

### Run TENT

Episodic TENT resets adapted weights after every batch:

```bash
python -u inference.py \
  datasets=gptsniffer_aigcodeset_test \
  tta=tent \
  tta.lr=1e-5 \
  tta.steps=1 \
  tta.reset_each_batch=true
```

Accumulated TENT keeps updates across batches:

```bash
python -u inference.py \
  datasets=gptsniffer_aigcodeset_test \
  tta=tent \
  tta.lr=1e-5 \
  tta.steps=1 \
  tta.reset_each_batch=false
```

### Run Centroid-Based TTA

```bash
python -u inference.py \
  datasets=gptsniffer_aigcodeset_test \
  tta=centroids \
  tta.distance=cosine \
  tta.logit_scale=10
```

### Run T2A-Inspired TTA

```bash
python -u inference.py \
  datasets=gptsniffer_aigcodeset_test \
  tta=t2a \
  tta.lr=1e-6 \
  tta.steps=1 \
  tta.reset_each_batch=true
```

## Dataset Config Names

Use one of these Hydra dataset configs:

```text
gptsniffer_python_test
gptsniffer_python_no_comment_test
gptsniffer_java_test
gptsniffer_aigcodeset_test
```

Example:

```bash
python -u inference.py datasets=gptsniffer_java_test tta=centroids
```

## Logging

WandB is enabled by default in `src/configs/inference_gptsniffer.yaml`.

To set project and run names:

```bash
python -u inference.py \
  datasets=gptsniffer_java_test \
  tta=centroids \
  writer=wandb \
  writer.project_name=gptsniffer-tta-detection \
  writer.run_name=java_centroids
```

To disable logging:

```bash
writer.mode=disabled
```

Comet ML can be used with:

```bash
writer=cometml
```

## Reproducing Main Experiments

The Colab notebooks automate the main runs:

- `run_main.ipynb`: four target datasets with `none`, `tent`, accumulated `tent`, and `centroids`.
- `run_t2a.ipynb`: T2A-inspired runs on all four datasets.
- `run_aigcodeset_hyperparameters.ipynb`: AIGCodeSet ablations for TENT and centroid settings.
- `run_codegendetect_codebert.ipynb`: evaluation of the external CodeBERT detector.
- `run_gptsniffer_diagnostics.ipynb`: prediction-bias diagnostics.

The notebooks assume a Colab-style environment and a local dataset archive placed
under `gptsniffer_finetuning/dataset.zip`.

## Extracting Results

The repository includes helper scripts for extracting logged metrics:

```bash
python extract.py
python extract_hyperparameters.py
python extract_t2a.py
python extract_wandb_plots.py
```

Generated CSV files used in the report include:

```text
wandb_metrics.csv
wandb_hyperparameter_metrics.csv
wandb_t2a_metrics.csv
sklearn_logreg_metrics.csv
codegendetect_codebert_metrics.csv
gptsniffer_diagnostics.csv
```

Plots are saved under:

```text
wandb_plots/
```

## Sanity Checks

Run the TF-IDF logistic regression baseline:

```bash
python sklearn_logreg_baseline.py
```

Run the external CodeBERT detector:

```bash
python codegendetect_codebert_eval.py
```

Run source detector diagnostics:

```bash
python gptsniffer_diagnostics.py
```

These checks are useful for separating implementation issues from genuine target
distribution shift.

## Notes

- The default inference batch size is `128`, chosen for A100/Colab-style runs.
- For local CPU testing, reduce `dataloader.batch_size`.
- Local predictions are saved under `data/saved/<inferencer.save_path>` only when
  `inferencer.save_predictions=true`.
- TTA does not save adapted model weights; adaptation happens during inference.

## References

Main detector and model:

- GPTSniffer: [GPTSniffer: A CodeBERT-based classifier to detect source code written by ChatGPT](https://www.sciencedirect.com/science/article/pii/S0164121224001043).
- CodeBERT: [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155).

TTA and pseudo-labeling methods:

- TENT: [Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726).
- Pseudo-labeling: [Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks](https://citeseerx.ist.psu.edu/document?doi=798d9840d2439a0e5d47bcf5d164aa46d5e7dc26&repid=rep1&type=pdf).
- SHOT: [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.08546).
- T2A: [Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation](https://arxiv.org/abs/2505.18787).

Evaluation datasets and baselines:

- AIGCodeSet: [A new annotated dataset for AI generated code detection](https://huggingface.co/datasets/basakdemirok/AIGCodeSet).
- External detector: [CodeGenDetect-CodeBert](https://huggingface.co/azherali/CodeGenDetect-CodeBert).
- Project template: [Blinorot PyTorch project template](https://github.com/Blinorot/pytorch_project_template/tree/main).

## License

See [LICENSE](LICENSE).
