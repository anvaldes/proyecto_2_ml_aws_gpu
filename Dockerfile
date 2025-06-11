FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY requirements.txt .
COPY train.py .
COPY modelo_base/ ./modelo_base/
COPY config_accelerate.yaml .

# Crear carpeta de configuración de accelerate
RUN mkdir -p /root/.cache/huggingface/accelerate && \
    cp config_accelerate.yaml /root/.cache/huggingface/accelerate/default_config.yaml

# Instalar dependencias
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Entrypoint estándar para SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV TRANSFORMERS_CACHE=/tmp

ENTRYPOINT ["python", "train.py"]
