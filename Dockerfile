FROM python:3.12-slim

# opencv/rembg need these; the -headless wheels keep the list short.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the segmentation model into the image so the first request is not a
# 176 MB download.
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY app ./app

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
