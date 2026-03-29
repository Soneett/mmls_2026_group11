# Сжатие эмбеддингов в задаче ранжирования для рекомендательных систем

Этот репозиторий содержит реализацию **рекомендательной системы**, обучающей эмбеддинги пользователей и объектов на основе данных взаимодействий.

Реализован модуль **сжатия эмбеддингов**, который позволяет уменьшить размерность представлений при минимальной потере качества ранжирования.

Проект выполнен в рамках курса **Missing ML Semester (HSE, 2026)**.

## Структура проекта

```text
mmls_2026_group11
├── src
│   ├── lightning
│   │   ├── data.py
│   │   └── model.py
│   │
│   ├── dataset
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   ├── temporal_dataset.py
│   │   └── temporal_dataloader.py
│   │
│   ├── graph
│   │   └── graph_compose.py
│   │
│   ├── training
│   │   ├── runner.py
│   │   ├── train_epoch.py
│   │   ├── evaluation.py
│   │   └── state.py
│   │
│   ├── utils
│   │   └── seed.py
│   │
│   ├── config.py
│   └── train.py
│
├── models
│   ├── encoder.py
│   ├── compressor.py
│   └── decoder.py
│
├── configs
│   └── base.yaml
│
├── data
│   └── ml100k_ratings.csv
│
├── notebooks
│   ├── EDA_movielens100k.ipynb
│   └── prototype.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── Makefile
├── HISTORY.md
└── README.md
```

## Датасет

В экспериментах используется датасет **MovieLens-100K**.

Формат входных данных:

```text
user_id,item_id,rating,timestamp
```

Каждая строка соответствует одному взаимодействию пользователя с объектом.

Файл расположен в:

```text
data/ml100k_ratings.csv
```

## Архитектура модели

В работе рассматриваются две архитектуры.

### Базовая модель (без сжатия)

```text
input data → Backbone → Head_big → score
```

- **Backbone** строит эмбеддинги пользователей и объектов размерности **d**.
- **Head_big** вычисляет скор релевантности объекта для пользователя.

### Модель со сжатием

```text
input data → Backbone → Compressor → Head_small → score
```

После backbone добавляется модуль **compressor**, который проецирует эмбеддинги в пространство меньшей размерности **d'**, где

```text
d' << d
```

Цель обучения:

```text
min ΔQuality  при  d' << d
```

То есть уменьшить размерность эмбеддингов при минимальной потере качества рекомендаций.

## Основные компоненты

### Encoder

`models/encoder.py`

Модель, обучающая **эмбеддинги пользователей и объектов** на основе взаимодействий.

### Compressor

`models/compressor.py`

Нейросетевой модуль, который переводит эмбеддинги большой размерности в **компактное пространство**.

### Decoder

`models/decoder.py`

Вычисляет скор релевантности между эмбеддингами пользователя и объекта.

### Обработка данных

`src/dataset/`

Отвечает за:

- загрузку датасета
- предобработку данных
- формирование временных батчей взаимодействий

### Построение графа

`src/graph/graph_compose.py`

Формирует граф взаимодействий пользователей и объектов.

### Обучение модели

Основная логика обучения организована через **PyTorch Lightning**.

- `src/lightning/data.py` содержит `LightningDataModule` для подготовки train / val / test dataloader.
- `src/lightning/model.py` содержит `LightningModule` с логикой обучения, валидации и тестирования.
- `src/training/` содержит вспомогательные функции для вычисления loss, метрик, инициализации optimizer и служебных объектов состояния.

## Запуск обучения

Обучение запускается из корня проекта:

```bash
python -m src.train
```

При запуске:
1. загружается конфиг из configs/base.yaml,
2. читается датасет data/ml100k_ratings.csv,
3. строятся временные срезы взаимодействий,
4. подготавливаются train / val / test dataloader,
5. запускается обучение модели через PyTorch Lightning,
6. сохраняются метрики и checkpoint лучшей модели.

## Ноутбуки

В папке `notebooks` находятся вспомогательные ноутбуки:

- **EDA_movielens100k.ipynb** - разведочный анализ данных
- **prototype.ipynb** - прототип модели

### Запуск в Docker

Проект поддерживает запуск в изолированном окружении с использованием Docker.

Все команды необходимо выполнять из корня проекта.

## Сборка контейнера
```bash
docker-compose build
```
## Запуск обучения
```bash
docker-compose up --build
```

При запуске контейнер автоматически:

1. устанавливает зависимости
2. загружает данные (из папки data/)
3. запускает обучение (src.train)

## Остановка контейнера
```bash
docker-compose down
```
## Переменные окружения

Для логирования экспериментов используется **Weights & Biases**.

Необходимо передать WANDB_API_KEY.

Способ 1 (однократный запуск)
```bash
WANDB_API_KEY=your_wandb_api_key docker-compose up --build
```
Способ 2 (через .env файл)

1. Создайте файл .env в корне проекта:
```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
```

2. После этого выполните:
```bash
docker-compose up --build
```
Docker автоматически подгрузит переменные из .env.

