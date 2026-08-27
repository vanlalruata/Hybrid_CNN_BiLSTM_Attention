# Hybrid CNN-BiLSTM Attention Framework for Cross-Dataset DDoS Detection in IoT Networks with Explainable AI

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

Abstract
With the swift growth of Internet of Things (IoT) networks, the susceptibility of connected devices to Distributed Denial-of-Service (DDoS) attacks has markedly risen. Traditional intrusion detection systems (IDS) and conventional machine learning approaches often fail to generalise across heterogeneous IoT environments due to their inability to adapt to varied feature spaces, and they lack the capacity to capture the complex spatial–temporal patterns inherent in network traffic data. To address these critical limitations, this paper proposes a novel, interpretable hybrid deep learning framework that integrates Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and a dynamic attention mechanism for robust, cross-dataset IoT DDoS detection. The novelty of the proposed framework lies in: (i) a parallel dual-branch spatial-temporal extraction mechanism that concurrently models localised packet header dependencies and sequential traffic dynamics; (ii) a self-attention mechanism that dynamically refines features by prioritising key malicious signatures; (iii) a unified feature harmonisation strategy that maps heterogeneous network protocols and feature configurations into a standardised format, enabling generalisation on unseen datasets; and (iv) a multi-level explainable AI (XAI) framework integrating SHAP, LIME, and ANOVA to provide joint statistical, global, and local explanations of model decisions. The framework is evaluated on four public benchmark IoT datasets, namely Apose-IoT-23, BoT-IoT, CIC-IoMT-2024, and CIC-IoT-2025, under both binary and multiclass classification settings. Experimental results demonstrate exceptional detection performance; under binary classification, the framework achieves accuracies of 99.31% on Apose-IoT-23, 99.998% on BoT-IoT, 99.83% on CIC-IoMT-2024, and a flawless 100.00% on CIC-IoT-2025. In the more challenging multiclass scenarios, the model maintains high robustness, achieving classification accuracies of 98.44% (Apose-IoT-23), 99.72% (BoT-IoT), 99.24% (CIC-IoMT-2024), and 99.997% (CIC-IoT-2025). Cross-dataset evaluation further confirms the model’s generalisation capability, with minimal performance degradation on unseen datasets. In addition, the proposed framework incorporates XAI techniques, including SHAP, LIME, and ANOVA, to provide both global and local interpretability of model predictions. Efficiency analysis demonstrates that the approach maintains low latency and energy proxies, rendering it suitable for real-time edge-based IoT deployment. Overall, the proposed hybrid CNN--BiLSTM-attention framework offers a robust, generalisable, and interpretable solution for DDoS detection in heterogeneous IoT environments.

## How to cite

Vanlalruata Hnamte, Hybrid CNN-BiLSTM Attention Framework for Cross-Dataset DDoS Detection in IoT Networks with Explainable AI, Volume 1, Issue 1, 2026, 100248, ISSN 3051-0643, <https://doi.org/> (<https://www.sciencedirect.com/science/article/pii/>)

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
