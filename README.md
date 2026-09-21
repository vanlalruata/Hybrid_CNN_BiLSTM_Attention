# Cross-Dataset Zero-Shot IoT Attack Detection Using a Hybrid CNN-BiLSTM+Attention Framework with Explainable AI

## Journal

Measurement: Digitalization

## Publisher

Elsevier

## Corresponding author

Vanlalruata Hnamte

## First author

Vanlalruata Hnamte

## DOI

<https://doi.org/>

ScienceDirect Link
<https://www.sciencedirect.com/science/article/pii/>

## Abstract

The rapid proliferation of Internet of Things (IoT) devices has heightened network susceptibility to diverse cyber-attacks and volumetric threats. Conventional intrusion detection systems (IDS) often struggle to generalise across heterogeneous network environments due to unstandardised feature spaces, distribution shifts, and an inability to capture complementary spatial and recurrent feature representations. To address these challenges, this paper proposes an attention-gated hybrid CNN--BiLSTM deep learning framework for robust, interpretable cross-dataset zero-shot IoT attack detection. The architecture integrates parallel 1D-CNN spatial filtering and BiLSTM feature-channel encoding, an attention-based feature-gating layer for dynamic representation refinement, a common preprocessing protocol with dataset-specific feature representations for leak-free flow representations, and a multi-level explainable AI (XAI) framework combining SHAP, LIME, and ANOVA. Evaluated across five independent random seeds on four benchmark datasets (Apose-IoT-23, BoT-IoT, CIC-IoMT-2024, and CIC-IoT-2025) under a leakage-free 70%:15%:15% split, the framework achieves mean binary classification accuracies of 99.33%, 99.997%, 99.83\%, and 98.73%, with macro-averaged F1-scores of 99.15%, 94.68%, 97.90%, and 98.70%, respectively. In multiclass threat scenarios, the model maintains high accuracy (98.86%, 99.74%, 99.21%, and 97.61%) and macro F1-scores up to 96.29\%. Furthermore, zero-shot cross-dataset evaluation assesses out-of-distribution transferability across heterogeneous IoT topologies and protocol stacks using a deterministic dimensional adaptation operator. Tiered XAI interpretations provide complementary global and local feature attribution transparency, while GPU efficiency benchmarking demonstrates low amortised per-sample batch inference latencies (0.0123--0.0160~ms/sample), establishing a solid empirical foundation for future hardware-based edge security deployment.

## How to cite

Vanlalruata Hnamte, Cross-Dataset Zero-Shot IoT Attack Detection Using a Hybrid CNN-BiLSTM+Attention Framework with Explainable AI, Volume 1, Issue 1, 2026, 100248, ISSN 3051-0643, <https://doi.org/> (<https://www.sciencedirect.com/science/article/pii/>)

## Note

If you find this code and paper useful, kindly consider to cite from your valuable work.

## How to use

## Interactive (menu)

```
python main.py
```

Follow options 1–9.

```
==================== IDS EXPERIMENT CLI ====================
(1) Load & split dataset (persist)
(2) Load existing split (validate)
(3) Visualize features (EPS)
(4) Train model (save ckpt + curves + history CSV)
(5) Evaluate & plot performance (EPS + CSV/JSON)
(6) XAI (SHAP/LIME/ANOVA)
(7) Ablation (user-selected feature subset)
(8) Evaluate (explicit)
(9) Inference (load_model → predict live CSV)
(q) Quit
============================================================
```

## Automation (examples)

* Split raw dataset:

```
python main.py split --dataset EDGE_IIoT --class_type binary --test_size 0.2 --seed 42
```

* Train on an existing split:

```
python main.py train --split_key EDGE_IIoT_binary_2025-10-29_08-30-00 --model fusion --epochs 60 --batch 256
```

* Evaluate:

```
python main.py eval --split_key EDGE_IIoT_binary_2025-10-29_08-30-00
```

* XAI:

```
python main.py xai --split_key EDGE_IIoT_binary_2025-10-29_08-30-00 --methods shap,lime,anova
```

* Ablation (user-selected):

```
python main.py ablate --split_key EDGE_IIoT_binary_2025-10-29_08-30-00 \
  --keep_feats "Duration,Rate,Srate,Drate,syn_flag_number,ack_flag_number"
```

* Inference:

```
python main.py infer --split_key EDGE_IIoT_binary_2025-10-29_08-30-00 --csv data/new/live_batch.csv
```
