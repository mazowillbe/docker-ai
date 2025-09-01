FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face token as env variable
ENV HUGGINGFACE_HUB_TOKEN=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

# Pre-download Qwen-3 1.7B instruct model
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='Qwen/Qwen-3-1.7B-Instruct', use_auth_token=True)"

# Copy project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
