# MULTIPINN

MULTIPINN — фреймворк для Physics-Informed Neural Networks (PINNs) на Python и PyTorch.

## Основная инструкция

Каноническая инструкция по установке и первому запуску находится в файле:

- [`guide/getting_started.ipynb`](guide/getting-started.ipynb)

Все пользовательские инструкции в репозитории должны использовать единый базовый вариант:

- **Поддерживаемые версии Python:** `3.8` – `3.11`
- **Рекомендуемая версия для нового окружения:** `3.10`
- **Основная команда установки:** `pip install -e .`

## Быстрый старт

```bash
git clone https://github.com/multipinn/multipinn.git
cd multipinn
python -m pip install --upgrade pip
pip install -e .
python -m examples.poisson_2D_1C.run_train
```

## Разделы документации

- **Руководство пользователя** — `guide/getting-started.ipynb`
- **Примеры** — каталог `examples/`
- **Исходный код** — пакет `multipinn/`
- **Тесты** — каталог `tests/`
