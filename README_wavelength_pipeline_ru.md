# Единый пайплайн сравнения методов отбора волн

Считает метрики и время работы для методов, которые остаются в дипломе:

- полный спектр + LR;
- Mutual Information top-12 + LR;
- PCA, 12 компонент + LR;
- SPA top-12 + LR;
- ADMM direct LS;
- ADMM top-12 + LR;
- автоэнкодер latent + LR.

Во всех задачах используется групповое разбиение по растениям: одно растение не попадает одновременно в train и test.

## Выходы одного запуска

В каждой папке результата будут:

- `method_metrics.csv` — Accuracy, Balanced Accuracy, F1 macro, число признаков, время;
- `selected_wavelengths.csv` — выбранные волны для MI, SPA, ADMM top-12;
- `run_summary.json` — размер выборки, split, plant leakage;
- `00_results_table_ru.png` — таблица метрик;
- `01_balanced_accuracy_ru.png` — график BA;
- `02_runtime_log_ru.png` — график времени;
- `03_accuracy_ba_f1_ru.png` — сравнение Accuracy/BA/F1;
- `04_selected_wavelengths_ru.png` — выбранные волны на средних спектрах;
- `05_confusion_best_ru.png` — confusion matrix лучшего метода;
- `06_admm_residuals_ru.png`, `07_admm_objective_ru.png` — графики ADMM;
- `08_autoencoder_loss_ru.png` — обучение автоэнкодера.

## Команды запуска

### FX17, здоровые / больные

```bash
python3 /beta/home/sashanoir/run_wavelength_pipeline_ru.py \
  --input-dir /beta/home/sashanoir/plots_fx_plantday_12masks \
  --outdir /beta/home/sashanoir/wavecmp_fx_health_drop_day9 \
  --dataset-name "FX17" \
  --task health \
  --drop-days 9 \
  --k-waves 12 \
  --device cuda \
  --admm-max-iter 2000 \
  --admm-tol 1e-4 \
  --ae-epochs 300
```

### FX17, временные стадии

```bash
python3 /beta/home/sashanoir/run_wavelength_pipeline_ru.py \
  --input-dir /beta/home/sashanoir/plots_fx_plantday_12masks \
  --outdir /beta/home/sashanoir/wavecmp_fx_stages_drop_day9 \
  --dataset-name "FX17" \
  --task stages \
  --drop-days 9 \
  --stage-map "ранняя:0,1,2,3,4,5;переходная:6,7,8;поздняя:10,11;симптоматическая:12" \
  --k-waves 12 \
  --device cuda \
  --admm-max-iter 2000 \
  --admm-tol 1e-4 \
  --ae-epochs 300
```

### IQ, здоровые / больные

```bash
python3 /beta/home/sashanoir/run_wavelength_pipeline_ru.py \
  --input-dir /beta/home/sashanoir/plots_iq_plantday_clean \
  --outdir /beta/home/sashanoir/wavecmp_iq_health_drop_day0_1_2 \
  --dataset-name "IQ" \
  --task health \
  --drop-days 0 1 2 \
  --k-waves 12 \
  --device cuda \
  --admm-max-iter 2000 \
  --admm-tol 1e-4 \
  --ae-epochs 300
```

### IQ, временные стадии

```bash
python3 /beta/home/sashanoir/run_wavelength_pipeline_ru.py \
  --input-dir /beta/home/sashanoir/plots_iq_plantday_clean \
  --outdir /beta/home/sashanoir/wavecmp_iq_stages_drop_day0_1_2 \
  --dataset-name "IQ" \
  --task stages \
  --drop-days 0 1 2 \
  --stage-map "ранняя:0,1,2,3;переходная:4;поздняя:5,7,8,9;симптоматическая:10" \
  --k-waves 12 \
  --device cuda \
  --admm-max-iter 2000 \
  --admm-tol 1e-4 \
  --ae-epochs 300
```

## Сводные графики по четырем запускам

```bash
python3 /beta/home/sashanoir/summarize_wavelength_runs_ru.py \
  --run-dirs \
    /beta/home/sashanoir/wavecmp_fx_health_drop_day9 \
    /beta/home/sashanoir/wavecmp_fx_stages_drop_day9 \
    /beta/home/sashanoir/wavecmp_iq_health_drop_day0_1_2 \
    /beta/home/sashanoir/wavecmp_iq_stages_drop_day0_1_2 \
  --outdir /beta/home/sashanoir/wavecmp_summary_ru
```

## Быстрый запуск без автоэнкодера

```bash
--methods full mi pca spa admm_direct admm_top12
```

## Быстрый запуск без ADMM

```bash
--methods full mi pca spa autoencoder
```
