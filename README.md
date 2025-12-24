````markdown
# TextCNN для анализа тональности (IMDb / Yelp)

В репозитории есть несколько моделей (RNN, LSTM, CNN), но этот README посвящён **только CNN-подходу** к классификации отзывов.

---

## 1. Как работает BPE-токенизатор «на пальцах»

Идея: вместо того чтобы хранить в словаре _все слова целиком_, мы учимся разбирать слова на **часто встречающиеся кусочки** (subword-ы).

Представь:

- Сначала мы считаем, что **слово — это просто набор символов** + специальный символ конца слова `</w>`.
  - Например:
    `movie</w> → m o v i e </w>`

- Затем мы смотрим на все тексты и считаем, **какие пары символов встречаются рядом чаще всего**.
  - Допустим, чаще всего встречается пара `m o`, потом `mo v`, потом `ov i` и т.п.

- На каждом шаге:
  1. Берём самую частую пару символов, например `m` + `o`.
  2. **Склеиваем её в один токен**: `mo`.
  3. Пробегаемся по всем словам и заменяем `m o` на `mo`.
  4. Обновляем частоты пар (потому что структура слов изменилась).
  5. Повторяем, пока словарь не вырастет до нужного размера `vocab_size`.

В итоге:

- Слова из текста **разбиваются на последовательность subword-токенов**:
  - `fantastic</w>` → `fan`, `tas`, `tic`, `</w>`
  - `fantasy</w>` → `fan`, `ta`, `sy`, `</w>`
- Часто встречающиеся кусочки (приставки, корни, суффиксы) попадают в словарь, а редкие слова всё равно можно разобрать на куски.

При использовании токенайзера (`bpe_tokenizer.py`):

- `encode(text, return_attention_mask=True)`:
  - разбивает текст на слова;
  - каждое слово прогоняет через BPE-разбиение;
  - заменяет subword-ы на их id из словаря;
  - выдаёт `input_ids`;
  - опционально строит `attention_mask` (1 — реальный токен, 0 — паддинг).

- `decode(ids)`:
  - берёт id → токены → склеивает обратно в слова, удаляя служебный токен `</w>`.

Главная польза:

- Мы не взрываемся от огромного словаря «все слова мира»;
- Умеем обрабатывать **редкие и новые слова**, которых модель не видела на обучении.

---

## 2. Как работает TextCNN «на пальцах»

TextCNN — это свёрточная нейросеть, только не для картинок, а для текста (`CNN.py`).

1. **Embedding-слой**
   - Каждый токен id → вектор фиксированной длины (например, 100/128).
   - Получаем матрицу размером:
     `batch_size × seq_len × embedding_dim`.

2. **Свёртки (Conv1d) с разными «ширинами окна»**
   - В коде используется несколько свёрток с `kernel_size = 3, 4, 5`.
   - Можно думать так:
     - свёртка с размером 3 «смотрит» на каждые **3 соседних токена** → улавливает короткие фразы;
     - размер 4 — на 4 токена подряд;
     - размер 5 — на 5 токенов.

   То есть каждая свёртка учится распознавать «паттерны» в тексте:
   - «очень хороший фильм»
   - «ужасная игра актёров»
   - «не рекомендую никому»
     и т.п.

3. **ReLU + Max Pooling**
   - После свёртки применяем `ReLU`, затем **максимум по времени** (`max_pool1d`).
   - Для каждого фильтра берём одно число — насколько сильно этот фильтр где-то сработал в тексте.
   - Это похоже на:
     > «мы пробежались по всему тексту, посмотрели, где шаблон сработал сильнее всего, и запомнили эту максимальную реакцию».

4. **Конкатенация + полносвязный слой**
   - Объединяем все максимумы от всех фильтров (разных размеров окна).
   - Применяем `dropout`, чтобы не переобучиться.
   - Прогоняем через `Linear` → получаем логиты классов (например, 2 класса: `negative`, `positive`).

Итог: TextCNN смотрит на текст как на «картинку 1×N», скользит фильтрами по словам и ловит важные локальные фразы, определяющие тональность.

---

## 3. Описание датасетов

### IMDb

Классический датасет отзывов к фильмам:

- Отзывы на английском языке;
- Каждый отзыв помечен:
  - `0` — **negative** (плохой отзыв),
  - `1` — **positive** (хороший отзыв).

В этом проекте данные лежат так:

```text
data/
  IMDb/
    train/
      positive.txt   # по одному отзыву в строке (метка 1)
      negative.txt   # по одному отзыву в строке (метка 0)
    test/
      positive.txt
      negative.txt
```
````

Класс `IMDBDataset` (`dataset.py`):

- читает указанные файлы `positive_path` и `negative_path`;
- хранит список `(текст, метка)`;
- при `__getitem__`:
  - токенизирует текст через `BPETokenizer`;
  - обрезает / дополняет до `max_length`;
  - возвращает:
    - `input_ids` — индексы токенов;
    - `attention_mask` — 1 для реальных токенов, 0 для паддинга;
    - `labels` — 0 или 1.

### Yelp

В папке `data/Yelp/` лежат отзывы из другого домена (рестораны / сервисы и т.п.):

```text
data/
  Yelp/
    reviews.txt   # по одному отзыву в строке, без меток
```

Часто такие данные используют:

- для дополнительного обучения токенизатора (чтобы словарь знал больше слов и выражений);
- как альтернативный источник текстов (например, для предобучения эмбеддингов).

---

## 4. Инструкция по обучению CNN (из консоли)

### 4.1. Подготовка окружения (uv)

```bash
# Клонируем/распаковываем проект
cd mlops-project

# Устанавливаем uv (если ещё нет): https://docs.astral.sh/uv/getting-started/installation/

# Синхронизируем зависимости по lock-файлу
uv sync --locked

# Для Docker зависимости ставятся через uv.export из lock-файла
```

Убедись, что данные лежат как ожидается:

```text
data/
  IMDb/
    train/
      positive.txt
      negative.txt
    test/
      positive.txt
      negative.txt
  Yelp/
    reviews.txt   # если используется
```

### 4.2. Обучение из консоли через `train_cnn.py`

В репозитории есть скрипт `train_cnn.py`, который:

- обучает BPE-токенизатор на `data/IMDb/train/*`;
- создаёт `IMDBDataset` для train/test;
- обучает `TextCNN`;
- сохраняет веса модели и токенизатор.

Конфиги Hydra лежат в `configs/` и управляют путями данных, гиперпараметрами, MLflow и т.п. Всё можно переопределять из CLI в стиле `param=value`.

Ключевые файлы:

- `configs/data/imdb.yaml` — пути к train/test и `max_length`
- `configs/tokenizer/bpe.yaml` — `vocab_size`, `num_merges`
- `configs/model/textcnn.yaml` — `embedding_dim`, `num_filters`, `dropout`, `num_classes`
- `configs/train/default.yaml` — `epochs`, `batch_size`, `lr`
- `configs/mlflow/default.yaml` — `enabled`, `mlflow_uri`, `experiment`, `run_name`
- `configs/paths/default.yaml` — пути до чекпоинта и токенизатора

Запуск с настройками по умолчанию:

```bash
uv run python train_cnn.py
```

Примеры переопределений:

```bash
# 5 эпох, другой learning rate и max_length
uv run python train_cnn.py train.epochs=5 train.lr=5e-4 data.max_length=256

# изменить число BPE-слияний
uv run python train_cnn.py tokenizer.num_merges=2000

# логирование в MLflow
uv run python train_cnn.py mlflow.enabled=true mlflow.mlflow_uri=http://localhost:5000
```

Примечания:

- Если `artifacts/tokenizer.json` уже существует, `train_cnn.py` загрузит его и не будет заново обучать BPE (экономит время). Если его нет, токенайзер обучится и будет сохранён.
- Обучение токенайзера с большим `num_merges` может быть долгим; для быстрой пробы можно уменьшить до 1000–2000.
- Данные: используем Kaggle IMDB (50k). Скачай CSV `IMDB Dataset.csv` в `data/kaggle_imdb/` и запусти:
  ```bash
  uv run python prepare_imdb_kaggle.py
  ```
  Это сформирует `data/IMDb/train|test/{positive,negative}.txt`. Далее можно `dvc add data/IMDb && dvc push` (remote см. ниже).

---

## 5. Инструкция по запуску обученной модели (инференс)

Ниже — запуск уже обученной модели на произвольном тексте **из консоли**.

### 5.1. Инференс через `infer_cnn.py`

Скрипт `infer_cnn.py`:

- загружает **сохранённый при обучении** токенизатор из `artifacts/tokenizer.json`;
- загружает веса `TextCNN` из чекпоинта;
- выводит класс (`negative`/`positive`) и вероятность для заданного текста.

Простейший запуск:

```bash
uv run python infer_cnn.py text="The movie was surprisingly good and I really enjoyed it!"
```

Важно:

- перед этим должен быть запущен `train_cnn.py`, чтобы существовали:
  - чекпоинт модели `checkpoints/textcnn_imdb.pt`;
  - токенизатор `artifacts/tokenizer.json`;
- аргумент `--max-length` должен совпадать с тем, что использовался при обучении.

Примеры:

```bash
# другой путь к чекпоинту/токенайзатору
uv run python infer_cnn.py \
    paths.checkpoint_path=checkpoints/textcnn_imdb.pt \
    paths.tokenizer_path=artifacts/tokenizer.json \
    text="I hated this movie, it was awful."

# изменить max_length
uv run python infer_cnn.py data.max_length=256 text="Absolutely fantastic, would watch again!"
```

### 5.2. pre-commit (форматирование/линты)

В репозитории настроены хуки: базовые `pre-commit-hooks` (whitespace/EoF), `black`, `isort`, `flake8`, `prettier` (Markdown/JSON/YAML/TOML/HTML/CSS).

Установка и запуск:

```bash
# поставить pre-commit (пример для uv)
uv tool install pre-commit

# установить хуки
pre-commit install

# прогнать на всех файлах
pre-commit run -a

# запустить отдельные линтеры/форматтеры при необходимости
uv run black .
uv run isort .
uv run flake8 .
```

---

## 7. Данные, MinIO и DVC

### 7.1. Подготовка данных (Kaggle IMDB 50k)

1. Скачай `IMDB Dataset.csv` с Kaggle (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`) в `data/kaggle_imdb/`.
2. Конвертируй в txt:
   ```bash
   uv run python prepare_imdb_kaggle.py
   ```
   Получишь `data/IMDb/train|test/{positive,negative}.txt`.

### 7.2. DVC и MinIO

- В репо настроен DVC remote `minio` (`.dvc/config`): `s3://mlops-data`, endpoint `http://minio:9000`, без SSL.
- В `docker-compose.yml` поднимаются:
  - `minio` (9000 API, 9001 консоль) c кредами `minio/minio123`;
  - `minio-mc`, который создаёт bucket и, если на хосте есть `./data/IMDb`, заливает его в `minio/mlops-data/imdb`.
- Чтобы загрузить данные в remote вручную:
  ```bash
  dvc add data/IMDb
  dvc push  # креды и endpoint берутся из .dvc/config
  git add data/IMDb.dvc .dvc/config .gitignore
  git commit -m "add imdb data via dvc"
  ```
- `download_data()` в `train_cnn.py`/`infer_cnn.py` пытается стянуть `data/IMDb` через `dvc.api.get`/`dvc pull`. Если данных нет в DVC, но есть Kaggle CSV, соберёт локально.

### 7.3. Docker и MLflow UI

- Запуск сервисов: `docker compose up --build`
- MLflow: `http://localhost:8080`
- MinIO: API `http://localhost:9000`, консоль `http://localhost:9001` (логин/пароль `minio/minio123`)
- Обучение в контейнере:
  ```bash
  docker compose exec app bash
  python train_cnn.py mlflow.enabled=true mlflow.mlflow_uri=http://mlflow:8080
  ```
  (Overrides Hydra так же, как в локальном запуске.)

### 7.1. Docker Compose

В репозитории есть `docker-compose.yml` с двумя сервисами:

- `mlflow` — UI и трекинг-сервер (порт 5000, стор — `sqlite:///mlflow.db`, артефакты — `./mlruns`);
- `app` — образ проекта (по умолчанию просто держит контейнер живым, чтобы можно было выполнять команды внутри).

Запуск:

```bash
docker compose up -d mlflow
```

MLflow UI будет доступен на `http://localhost:5000`.

### 7.2. Обучение в контейнере с подключением к MLflow

```bash
# поднять сервисы
docker compose up -d

# зайти в контейнер app
docker compose exec app bash

# внутри контейнера: запустить обучение с логированием в MLflow
python train_cnn.py --mlflow-uri http://mlflow:5000

# инференс (использует сохранённые артефакты в текущей директории)
python infer_cnn.py --text "Great movie!" --tokenizer-path artifacts/tokenizer.json
```

Важно: `--mlflow-uri` должен указывать на сервис `mlflow` из compose (`http://mlflow:5000`), а артефакты/чекпоинты сохраняются в общую volume (`.` смонтирована в `/app`).

---

## 6. Зависимости проекта

Основные зависимости (заданы в `pyproject.toml`, для uv ставятся через `uv sync`):

```txt
torch>=2.0.0
mlflow>=2.10.0
pytorch-lightning>=2.3.0
```

Плюс стандартная библиотека Python (`os`, `json`, `random` и т.д.), которая отдельно не ставится.

Для воспроизводимости используется `uv.lock`; устанавливайте зависимости с `uv sync --locked` или через Dockerfile, который ставит пакеты по lock-файлу.

````

## 8. Локальный запуск (без Docker)

1) Подготовить окружение:
```bash
uv sync --locked
````

2. Подготовить данные:

- Вариант A (DVC + MinIO): убедись, что MinIO доступен (endpoint `http://localhost:9000`, бакет `mlops-data`), экспортируй креды:
  ```bash
  export AWS_ACCESS_KEY_ID=minio
  export AWS_SECRET_ACCESS_KEY=minio123
  dvc remote modify minio endpointurl http://localhost:9000
  dvc pull
  ```
- Вариант B (без DVC): скачай Kaggle CSV `IMDB Dataset.csv` в `data/kaggle_imdb/` и собери txt:
  ```bash
  uv run python prepare_imdb_kaggle.py
  ```

3. Обучение:

```bash
uv run python train_cnn.py
# пример с MLflow
uv run python train_cnn.py mlflow.enabled=true mlflow.mlflow_uri=http://localhost:8080
```

4. Инференс:

```bash
uv run python infer_cnn.py 'text="Great movie!"'
# или с явными путями
uv run python infer_cnn.py \
  paths.checkpoint_path=checkpoints/textcnn_imdb.pt \
  paths.tokenizer_path=artifacts/tokenizer.json \
  'text="Great movie!"'
```

## 9. Экспорт в ONNX и TensorRT (production packaging)

### 9.1. Экспорт в ONNX

1. Поставь доп. зависимости (один раз):

```bash
uv pip install onnx onnxruntime onnxscript
```

2. Выполни экспорт (статическая длина = `data.max_length`, можно переопределить):

```bash
uv run python export_onnx.py \
  paths.checkpoint_path=checkpoints/textcnn_imdb.pt \
  paths.tokenizer_path=artifacts/tokenizer.json \
  paths.onnx_path=artifacts/textcnn.onnx \
  data.max_length=128
```

Скрипт проверит модель через `onnx.checker` и сделает пробный прогон в `onnxruntime`. 3) (Опционально) сохранить в DVC:

```bash
dvc add artifacts/textcnn.onnx
dvc push
```

### 9.2. Экспорт в TensorRT

- Нужен установленный TensorRT и доступная утилита `trtexec` (GPU-окружение).
- ONNX-файл должен уже существовать (см. пункт 9.1).

Запуск:

```bash
uv run python export_tensorrt.py paths.onnx_path=artifacts/textcnn.onnx paths.trt_path=artifacts/textcnn.trt data.max_length=128   # при желании переопределить max_length
```

Скрипт вызывает `trtexec` c динамическими размерами входа и сохраняет движок в `artifacts/textcnn.trt`.
Далее можно добавить его в DVC:

```bash
dvc add artifacts/textcnn.trt
dvc push
```

## 10. Triton Inference Server (ONNX, CPU)

> Для учебных целей можно поднять Triton в Docker на CPU. На macOS (Apple Silicon) будет медленно через эмуляцию `linux/amd64`. Для продакшн/GPU лучше использовать Linux c NVIDIA.

1. Подготовь модельный репозиторий (ONNX уже должен быть собран):

```bash
mkdir -p model-repo/textcnn/1
cp artifacts/textcnn.onnx model-repo/textcnn/1/model.onnx
# config.pbtxt уже лежит в model-repo/textcnn/config.pbtxt
```

2. Запусти Triton (Docker, CPU, уже добавлен в docker-compose):

```bash
# если нужен образ под amd64 на Mac:
# DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up triton
docker compose up triton
```

Порты: 8000 (HTTP), 8001 (gRPC), 8002 (metrics).
Проверка:

```bash
curl -s http://localhost:8000/v2/health/ready
curl -s http://localhost:8000/v2/models/textcnn
```

3. Пример HTTP-запроса на инференс:

```bash
python - <<'PY'
import json, requests
from bpe_tokenizer import BPETokenizer

TEXT = "Great movie!"
MAX_LEN = 128
tok = BPETokenizer.load("artifacts/tokenizer.json")
ids = tok.encode(TEXT)["input_ids"][:MAX_LEN]
pad = tok.pad_token_id
ids = ids + [pad]*(MAX_LEN-len(ids))

payload = {
    "inputs": [{
        "name": "input_ids",
        "shape": [1, MAX_LEN],
        "datatype": "INT64",
        "data": ids
    }],
    "outputs": [{"name": "logits"}],
}
resp = requests.post("http://localhost:8000/v2/models/textcnn/infer", json=payload)
resp.raise_for_status()
print(resp.json())
PY
```

Ответ содержит логиты; дальше выбираем argmax.

Альтернатива через curl (если ids уже подготовлены и паддированы до `MAX_LEN`):

```bash
curl -X POST http://localhost:8000/v2/models/textcnn/infer \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [{
            "name": "input_ids",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": [/* сюда подставь массив из 128 токенов */]
        }],
        "outputs": [{"name": "logits"}]
      }'
```

`input_ids` нужно получить заранее через `tokenizer.encode(text)` и допаддить до `data.max_length`.

4. (Опционально) добавь `model-repo/textcnn/1/model.onnx` в DVC:

```bash
dvc add model-repo/textcnn/1/model.onnx
dvc push
```

5. Быстрый клиент для Triton (HTTP)

В репозитории есть скрипт `triton_client.py`, который сам токенизирует текст и шлёт запрос в Triton.

```bash
uv run python triton_client.py \
  --text "Great movie!" \
  --tokenizer-path artifacts/tokenizer.json \
  --url http://localhost:8000/v2/models/textcnn/infer \
  --max-length 128
```

Он печатает ответ Triton и предсказанный класс. Если Triton работает не на localhost, поменяй `--url`.
