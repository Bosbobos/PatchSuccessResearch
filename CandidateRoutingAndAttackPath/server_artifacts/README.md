# Server artifacts

В этой папке хранятся важные результаты вычислений с SR004. Серверные имена
переводятся в канонические локальные понятия согласно
`../SERVER_NAMING_MAP.md`.

Копируем: конфигурации, metadata, summaries, таблицы метрик, графики, небольшие
модели и split manifests. Не копируем повторно вычислимые activation caches,
датасет, `__pycache__`, `.pytest_cache` и временные checkpoints.
