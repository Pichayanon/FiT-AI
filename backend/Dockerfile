# syntax=docker/dockerfile:1.7

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.base.txt requirements.ml.txt /app/

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary -r /app/requirements.base.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary -r /app/requirements.ml.txt

COPY . /app/

EXPOSE 5050 5051 5052 5053

# Overridden per service in docker-compose.yml
CMD ["python", "-m", "wall_sit.wall_sit_streaming"]
