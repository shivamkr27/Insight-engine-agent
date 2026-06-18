FROM python:3.11-slim

WORKDIR /app

# System libs needed to compile some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first — saves ~1.3GB vs the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure persistent directories exist inside the container
RUN mkdir -p docs data chroma_db parent_store

EXPOSE 8000

CMD ["chainlit", "run", "ui/app.py", "--host", "0.0.0.0", "--port", "8000"]
