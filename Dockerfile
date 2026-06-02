FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first (editable install needs the actual package dirs)
COPY . .

# Install dependencies via pyproject.toml (no dev extras)
RUN pip install --no-cache-dir -e .

# HF Spaces runs as non-root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]