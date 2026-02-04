# Reproducing Experiments: Violence Detection + Context

Short instructions for running the pipeline. **Data is not included** in the repository — it must be downloaded separately.

---

## Datasets (where to download)

| Dataset | Where to download | Where to place |
|---------|-------------------|----------------|
| **UBI-Fights** | https://socia-lab.di.ubi.pt/EventDetection/ (direct link: https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip) | `datasets/UBI_Fights/` (with subdirs: `videos/fight/`, `videos/normal/`, `annotation/`, and file `test_videos.csv`) |
| **NTU CCTV-Fights** | https://rose1.ntu.edu.sg/dataset/cctvFights/ (access must be requested) | `datasets/CCTV_Fights/` (folders `mpeg-*` with `.mpeg` files and file `groundtruth.json`) |
| **ViT-S/16 (pre-trained)** | https://tfhub.dev/sayakpaul/vit_s16_fe/1 | `models/HubModels/vit_s16_fe_1/` |

Scripts assume the project root is a directory containing `datasets/`, `models/`, `checkpoints/`, and `results/`. Run all commands from that root.

---

## Environment and dependencies

```bash
cd /path/to/dip
python3.9 -m venv venv
source venv/bin/activate
pip install -r reproducibility/requirements.txt
```

---

## Run order

1. **Train baseline**
   - UBI: `python reproducibility/scripts/train_UBI_simple.py`
   - CCTV: `python reproducibility/scripts/train_CCTV_simple.py`

2. **Cross-dataset evaluation**
   - UBI→CCTV: `python reproducibility/scripts/eval_cross_dataset_UBI_to_CCTV.py`
   - CCTV→UBI: `python reproducibility/scripts/eval_cross_dataset_CCTV_to_UBI.py`

3. **Context features**
   - Basic: `python reproducibility/scripts/compute_context_features.py`
   - Extended: `python reproducibility/scripts/compute_context_features_extended.py`

4. **Late fusion**
   - `python reproducibility/scripts/train_context_fusion.py`

5. **Ablations**
   - `python reproducibility/scripts/ablation_analysis.py`

Outputs are written to `results/` and `checkpoints/`. Scripts use absolute paths in their CONFIG; adjust them if the project is located elsewhere.
