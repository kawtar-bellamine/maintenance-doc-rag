FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npx vite build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docrag/ docrag/
COPY app/ app/
COPY data/ data/
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && python -m docrag.corpus && python -m docrag.ingest_md
COPY --from=web /web/dist web/dist
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
