FROM python:3.9 AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY main.py .
COPY crop_disease_model_weights.pth .

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]