FROM python:3.11-slim

WORKDIR /app

# System deps for building native extensions (faiss-cpu, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies via pyproject.toml (no dev extras)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy app
COPY . .

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]