FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rca ./rca

EXPOSE 8000

CMD ["uvicorn", "rca.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
