FROM python:3.12-slim

# Environment configs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY backend/requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install python-multipart

COPY . .

EXPOSE 8000

# ✅ Command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
