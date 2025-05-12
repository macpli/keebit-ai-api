FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# Install CPU-only PyTorch and OpenAI CLIP
RUN pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install git+https://github.com/openai/CLIP.git

# Install other dependencies
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]