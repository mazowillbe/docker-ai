FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face token
ENV HUGGINGFACE_HUB_TOKEN=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

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
