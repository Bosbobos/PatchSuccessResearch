# Таблица соответствия серверного нейминга

Эта таблица связывает нейтральные имена в серверном workspace с исходными
понятиями локального исследования. На сервере используются термины в стиле
анализа представлений, multi-view/JEPA и восстановления латентных признаков.

## Каталоги и файлы

| Серверное имя | Локальное имя / смысл |
|---|---|
| `tactile_jepa_diagnostics` | корень текущего исследования |
| `representation_routing` | `CandidateRoutingAndAttackPath` |
| `latent_tracing` | `CausalTracingViaPatching` |
| `view_response_analysis` | `patch_success_analysis` |
| `multi_view_response.ipynb` | `19_MultiPatchASR.ipynb` |
| `multi_view_response.py` | `multi_patch_asr.py` |
| `assets/texture_a.png` | `data/0709_yolo_dpatch_1000.png` |
| `assets/texture_b.png` | `data/cls_patch.png` |
| `assets/texture_c.png` | `data/depatch.png` |
| `assets/texture_d.png` | `data/nap1500new.png` |
| `assets/perception_backbone.pt` | `yolo11s.pt` |
| `outputs/multi_view_response` | результаты multi-patch ASR |
| `representation_routing/distributed_surface_alignment.py` | распределённое обучение одного `ART RobustDPatch` на четырёх GPU |
| `outputs/distributed_surface_alignment` | checkpoints, история и итоговый патч распределённого `RobustDPatch` |

## Понятия и метрики

| Серверный термин | Исходное понятие |
|---|---|
| `reference view` | чистое изображение |
| `augmented view` | изображение с патчем |
| `stimulus tile` / `texture` | патч |
| `response shift` | воздействие атаки |
| `confidence attenuation` | падение confidence из-за патча |
| `attenuation event` | успешная атака по порогу confidence drop |
| `attenuation rate` | ASR |
| `complete response suppression` | полное скрытие target detection |
| `latent residual target` | teacher-компонента между clean и patched |
| `residual predictor` | student-модель |
| `token support estimator` | локализатор компоненты |
| `token relevance scorer` | ranker кандидатов/кластеров |
| `latent restoration` | защитное вычитание предсказанной компоненты |
| `restored response` | предсказание детектора после защиты |
| `response recovery` | восстановление детекции защитой |
| `collateral response change` | ухудшение на исходно видимых/чистых примерах |
| `unmatched proposal group` | прежняя группа hidden/no-IoU |
| `weak-response group` | прежняя группа hidden/low-confidence |
| `native stimulus size` | исходный размер каждого патча |
| `controlled 160 view` | все патчи, приведённые к 160×160 |

## Основные метрики

| Серверная колонка | Локальная колонка |
|---|---|
| `conf_reference` | `conf_clean` |
| `conf_augmented` | `conf_patch` |
| `attenuation` | `drop = conf_clean - conf_patch` |
| `attenuation_event` | `success` |
| `attenuation_rate` | `asr` |
| `complete_suppression` | `complete_hide` |
| `reference_bbox_area_frac` | `clean_bbox_area_frac` |
| `stimulus_area_frac` | `patch_area_frac` |

Важные серверные артефакты копируются обратно в локальный репозиторий без
переименования метрик обратно: эта таблица остаётся единой точкой декодирования.
