# ClassifierDetectorExperiments

Side-by-side notebooks for comparing cached classifier and detector patch-success metrics.

The notebooks avoid heavy recomputation. Classifier artifacts are read from
`classifier_experiments/outputs/classifier_patch_analysis`, detector artifacts
from `new_experiments/outputs/patch_success_analysis` or the nested historical
path `new_experiments/new_experiments/outputs/patch_success_analysis`.

Derived L1/L2 caches are saved next to the corresponding existing cache:
classifier results under `classifier_experiments/.../cache`, detector results
under `new_experiments/.../cache`.

