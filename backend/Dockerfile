FROM python:3.10-slim

WORKDIR /app

# Install system deps (if you add more libs, extend here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY backend/ /app/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    opencv-python-headless \
    mediapipe \
    numpy \
    torch \
    scikit-learn \
    joblib

# Default command is overridden per service in docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]

