# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar primero las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Luego copiar el resto del código
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
