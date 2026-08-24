FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Removed python3-venv since we use Conda)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    libpq-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# CUDA Paths & Architecture Flags
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

### --- CONDA ENV SETUP ---
# Define Miniconda installation path
ENV CONDA_DIR=/opt/conda

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Put conda on environmental path
ENV PATH=$CONDA_DIR/bin:$PATH

# Configure conda to use conda-forge and strictly avoid the default commercial channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the Conda environment with Python 3.10 named 'env_marcis_v1'
RUN conda create -n env_marcis_v1 python=3.10 -y

# Automatically point all subsequent calls to the Conda environment's binaries
ENV PATH=$CONDA_DIR/envs/env_marcis_v1/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/envs/env_marcis_v1/lib:$LD_LIBRARY_PATH
ENV CONDA_DEFAULT_ENV=env_marcis_v1
### --------------------

### API Dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi[standard] huggingface_hub[hf_xet] alembic

### AI Dependencies
# install specific PyTorch wheels inside the conda environment
RUN pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
RUN conda install "ffmpeg" -c conda-forge
RUN pip install torchcodec==0.10 --index-url=https://download.pytorch.org/whl/cu130
# Requirements install
RUN pip install nvidia-resiliency-ext==0.3.0 python-dotenv
RUN pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@95f92737c"
RUN pip install -U funasr
RUN pip install sqlmodel asyncpg aio-pika loguru miniopy-async

# Copy the rest of the application code
COPY . /app