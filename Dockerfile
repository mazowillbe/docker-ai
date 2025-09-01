FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen-3 1.7B instruct model to bake into image
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='Qwen/Qwen-3-1.7B-Instruct')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
