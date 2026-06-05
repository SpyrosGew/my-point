FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 \
        torchvision==0.19.1+cu121 \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "pointing_demo"]
