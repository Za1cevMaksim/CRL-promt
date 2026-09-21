# Классификация эмоций

Сравниваются MLE, маска префикса, путаемая метка, REINFORCE и совместная конфигурация на Qwen2.5-3B. Все варианты одного запуска получают общий probe-старт.

```bash
python experiments/classification/run.py
```

Конфигурации находятся в `configs/classification_*.yaml`, результаты сохраняются в `outputs/classification/`.

