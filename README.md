# IMDb Sentiment Classifier (TextCNN + BPE)

Кратко: классификация отзывов IMDb на `positive/negative` с самописным BPE-токенизатором и TextCNN. Управление гиперпараметрами через Hydra, обучение — PyTorch Lightning, артефакты/данные тянутся через DVC, логирование опционально в MLflow. Есть экспорт в ONNX и TensorRT, клиент для Triton.

## Постановка задачи

Цель проекта — построить модель бинарной классификации текстовых отзывов о фильмах из датасета IMDB. Необходимо определить эмоциональную окраску текста: позитивный или негативный отзыв. Такая система может использоваться для автоматического мониторинга пользовательского мнения, анализа удовлетворенности аудитории, а также фильтрации отзывов по тональности.

## Формат входных и выходных данных

Входные данные:

- Текстовый отзыв на английском языке.

- После предварительной обработки преобразуется в последовательность индексов токенов фиксированной длины (например, 200–300 токенов).

- Формат входа в модель: `tensor<int>[batch_size, max_len]`.

Выходные данные:

`softmax([neg, pos])` — распределение вероятностей для двух классов.

Формат выхода:

`tensor<float>[batch_size, 2]`.

## Метрики

Для бинарной классификации наиболее релевантны:

- Accuracy — основная метрика, так как классы в IMDB примерно сбалансированы.

- F1-score — позволяет учитывать баланс precision/recall.

- AUC-ROC — отражает качество разделения классов по вероятностям.

Ожидаемые значения:

- Accuracy: 0.84–0.90 для CNN-бейзлайна.

- F1-score: 0.85+

Такие значения typical для моделей CNN на IMDB и достижимы на ограниченном количестве параметров.

## Валидация и тест

- Разделение на выборки:
  - Тренировочная: 80%

  - Валидационная: 10%

  - Тестовая: 10%

- Для воспроизводимости:
  - Фиксация random seed (например, `42`) для NumPy, PyTorch и Python.

  - Детерминированные операции при возможности.

Стандартный датасет IMDB уже содержит заранее разделённые тренировочную и тестовую выборки (25k / 25k), поэтому внутри train создаётся дополнительный validation split.

## Датасеты

Основной датасет: IMDB Dataset of 50K Movie Reviews (больше 50 мб)

- Источник: [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

- Размер:
  - 50 000 текстов (25k train, 25k test)

  - Длина отзывов до нескольких сотен слов

- Особенности:
  - Сильно варьируется длина текста → нужна обрезка/паддинг.

  - Тексты на английском, встречаются редкие слова, опечатки, сленг.

- Возможные проблемы:
  - Много сложных грамматических конструкций.

  - Датасет старый (2011) → часть выражений может быть редкой в современных моделях.

  - Возможен "data leakage", если не проверять структуру директорий.

# Моделирование

## Бейзлайн

Простейшее решение:

- BM25 → Logistic Regression.

- Простая архитектура:
  - Представление текста как мешка слов.

  - Классификация линейной моделью.

Ожидаемая точность бейзлайна: 0.7–0.8.

## Основная модель

Модель: CNN для текстовой классификации

- Этапы:
  1. Embedding layer (обучаемые с нуля)

  2. 1D Convolution с несколькими фильтрами разной ширины (например 3,4,5)

  3. Global Max Pooling

  4. Полносвязный слой \+ Dropout

  5. Softmax для бинарной классификации

**Обучение:**

- Оптимизатор: Adam

- Learning rate: \~1e-3

- Loss: Crossentropy

- Batch size: 128

- Epochs: 5–10

Модель проста, производительна и часто превосходит LSTM при меньшем количестве параметров.

# Внедрение

REST API сервис (FastAPI). Бэкенд будет ходить в сервис с моделью, отправлять ей отзыв и ожидать предсказание. Деплой будет в Docker. Предполагается настройка мониторинга и логирования.

## Setup

- Требования: Python ≥3.10, uv, dvc.
- Установка зависимостей:
  ```bash
  uv sync --locked
  ```
- DVC:
  - Дефолтный remote `myremote` указывает на локальный путь `../remote` (см. `.dvc/config`). Там уже лежит `data/IMDb`.
  - Чтобы подтянуть данные: `dvc pull data/IMDb.dvc` (или просто запустить `train-cnn`/`infer-cnn` — они вызовут `download_data()`).
  - Альтернатива: поставить Kaggle CSV `data/kaggle_imdb/IMDB Dataset.csv` и собрать: `uv run prepare-imdb-kaggle`.
- Проверка структуры данных (после pull/подготовки):
  ```
  data/
    IMDb/
      train/{positive.txt,negative.txt}
      test/{positive.txt,negative.txt}
  ```

## Train

Запуск из корня репозитория (нужен доступ к `configs/`):

```bash
uv run train-cnn
```

Полезные параметры:

- Гиперпарметры: `uv run train-cnn train.epochs=5 train.lr=5e-4 data.max_length=256 tokenizer.num_merges=2000`.
- MLflow: `uv run train-cnn mlflow.enabled=true mlflow.mlflow_uri=http://localhost:8080`.

Выходы:

- Чекпоинт: `checkpoints/textcnn_imdb.pt`.
- Токенизатор: `artifacts/tokenizer.json`.
- Метрики: `loss`, `accuracy`, `f1`, `auc` на train/val. Логи — в консоли; при включённом MLflow — в указанном трекинг-сервере (графики метрик появятся там).

## Production preparation

- ONNX (нужны `onnx`, `onnxruntime`):
  ```bash
  uv run export-onnx \
    paths.checkpoint_path=checkpoints/textcnn_imdb.pt \
    paths.tokenizer_path=artifacts/tokenizer.json \
    paths.onnx_path=artifacts/textcnn.onnx \
    data.max_length=128
  ```
- TensorRT (нужен `trtexec` из TensorRT, обычно Linux+GPU):
  ```bash
  uv run export-tensorrt \
    paths.onnx_path=artifacts/textcnn.onnx \
    paths.trt_path=artifacts/textcnn.trt \
    data.max_length=128
  ```
- Triton: положить `artifacts/textcnn.onnx` в `model-repo/textcnn/1/model.onnx`, `config.pbtxt` уже есть. Запуск: `docker compose up triton`. Клиент ниже.
- Артефакты для поставки: `checkpoints/textcnn_imdb.pt`, `artifacts/tokenizer.json`, `artifacts/textcnn.onnx` (и при наличии `artifacts/textcnn.trt`), плюс `configs/` для воспроизводимых запусков.

## Infer

- Локально (CPU/MPS/GPU — как доступно):
  ```bash
  uv run infer-cnn text="The movie was surprisingly good and I really enjoyed it!"
  ```
  Переопределения путей: `uv run infer-cnn paths.checkpoint_path=checkpoints/textcnn_imdb.pt paths.tokenizer_path=artifacts/tokenizer.json data.max_length=128 text="Bad movie"`.
- Формат входа: сырая строка текста; внутри код токенизирует, паддит до `data.max_length`. Вывод: класс (`negative/positive`) и вероятность.
- Triton HTTP (когда сервер запущен и модель скопирована):
  ```bash
  uv run triton-client \
    --text "Great movie!" \
    --tokenizer-path artifacts/tokenizer.json \
    --url http://localhost:8000/v2/models/textcnn/infer \
    --max-length 128
  ```

## Docker Compose (MLflow / Triton)

- MLflow поднимается командой:
  ```bash
  docker compose up -d mlflow
  ```
  Используй при запуске обучения с логированием в MLflow (`mlflow.enabled=true mlflow.mlflow_uri=http://localhost:8080`).
- Triton поднимается командой:
  ```bash
  docker compose up triton
  ```
  Используй, когда уже есть ONNX в `model-repo/textcnn/1/model.onnx` и нужно обслуживать инференс через Triton (HTTP 8000 / gRPC 8001).
