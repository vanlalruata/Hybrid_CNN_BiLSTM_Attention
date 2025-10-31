# How to use
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