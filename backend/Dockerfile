# backend/Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc curl && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --user -r /tmp/requirements.txt

# final image
FROM python:3.11-slim
WORKDIR /app
ENV PATH=/root/.local/bin:
COPY --from=builder /root/.local /root/.local
COPY backend /app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 PORT=8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
