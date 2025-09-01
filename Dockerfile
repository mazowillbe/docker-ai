# ---- Base image ----
FROM python:3.10-slim

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Create non-root user ----
RUN useradd -m -u 1000 appuser

# ---- Working directory ----
WORKDIR /app

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy source code ----
COPY . .

# ---- Switch to non-root ----
USER appuser

# ---- Port is injected by Render ($PORT) ----
EXPOSE $PORT

# ---- Start the server ----
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
