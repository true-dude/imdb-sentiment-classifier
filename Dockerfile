# syntax=docker/dockerfile:1.6

FROM python:3.10-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

# Сначала зависимости
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Теперь код
COPY . .

# Финальный sync (если появились extras/optional)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

CMD ["python", "train_cnn.py"]
