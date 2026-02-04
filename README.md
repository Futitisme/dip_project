# Воспроизведение экспериментов: Violence Detection + Context

Краткая инструкция по запуску. Данные в репозиторий **не входят** — их нужно скачать отдельно.

---

## Датасеты (где скачать)

| Датасет | Где скачать | Куда положить |
|--------|-------------|----------------|
| **UBI-Fights** | https://socia-lab.di.ubi.pt/EventDetection/ (прямая ссылка: https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip) | `datasets/UBI_Fights/` (внутри: `videos/fight/`, `videos/normal/`, `annotation/`, `test_videos.csv`) |
| **NTU CCTV-Fights** | https://rose1.ntu.edu.sg/dataset/cctvFights/ (нужно запросить доступ) | `datasets/CCTV_Fights/` (папки `mpeg-*` с `.mpeg` и файл `groundtruth.json`) |
| **ViT-S/16 (pre-trained)** | https://tfhub.dev/sayakpaul/vit_s16_fe/1 | `models/HubModels/vit_s16_fe_1/` |

Скрипты ожидают, что корень проекта — каталог с подпапками `datasets/`, `models/`, `checkpoints/`, `results/`. Запуск — из этого корня.

---

## Окружение и зависимости

```bash
cd /path/to/dip
python3.9 -m venv venv
source venv/bin/activate
pip install -r reproducibility/requirements.txt
```

---

## Порядок запуска

1. **Обучение baseline**
   - UBI: `python reproducibility/scripts/train_UBI_simple.py`
   - CCTV: `python reproducibility/scripts/train_CCTV_simple.py`

2. **Cross-dataset оценка**
   - UBI→CCTV: `python reproducibility/scripts/eval_cross_dataset_UBI_to_CCTV.py`
   - CCTV→UBI: `python reproducibility/scripts/eval_cross_dataset_CCTV_to_UBI.py`

3. **Контекстные признаки**
   - Базовые: `python reproducibility/scripts/compute_context_features.py`
   - Расширенные: `python reproducibility/scripts/compute_context_features_extended.py`

4. **Late fusion**
   - `python reproducibility/scripts/train_context_fusion.py`

5. **Абляции**
   - `python reproducibility/scripts/ablation_analysis.py`

Результаты пишутся в `results/` и `checkpoints/`. В скриптах в CONFIG заданы абсолютные пути; при другом расположении проекта их нужно поправить.
