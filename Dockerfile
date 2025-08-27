# ---------- Stage 1: builder (has compilers/headers) ----------
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Build tooling + headers (removed in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ \
      python3 python3-pip python3-venv \
      libopenblas-dev libomp-dev \
      libspatialindex-dev unixodbc-dev \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make compilers discoverable for pip builds
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++

# Create venv and upgrade pip
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1

# Copy only requirements first for better layer caching
WORKDIR /tmp/build
COPY requirements.txt .

# Install Python deps into the venv (wheels get built here if needed)
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime (lean) ----------
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest
ARG DEBIAN_FRONTEND=noninteractive

# Add Microsoft repo (Ubuntu 24.04 / noble) via keyrings, then install runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/ubuntu/24.04/prod noble main" \
      > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
      unixodbc \
      postgresql-client sudo \
      msodbcsql18 mssql-tools18 \
      libopenblas0-openmp \
      libspatialindex6 \
      g++ libgomp1 python3.12-dev \
      git openssh-client \
      vim wget && \
    echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' > /etc/profile.d/mssql-tools.sh && \
    rm -rf /var/lib/apt/lists/*

# Make compilers discoverable for JIT at runtime
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++

# Make SQL tools available in non-login shells too
ENV PATH="/opt/mssql-tools18/bin:${PATH}"

# Copy only the ready venv from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user and dirs
RUN useradd -m -s /bin/bash appuser \
 && echo "appuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/appuser \
 && chmod 0440 /etc/sudoers.d/appuser \
 && mkdir -p /data/tmp /workspace /workspaces/conduit \
 && chown -R appuser:appuser /opt/venv /data/tmp /workspace /workspaces

# Keep Docker VHDX small + steer heavy caches to /data/tmp
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1 \
    PYTENSOR_FLAGS="floatX=float64,optimizer_excluding=constant_folding,cxx=/usr/bin/g++,compiledir=/data/tmp/pytensor" \
    XLA_FLAGS="--xla_persistent_cache_dir=/data/tmp/jax" \
    JAX_CACHE_DIR="/data/tmp/jax" \
    DASK_TEMPORARY_DIRECTORY="/data/tmp/dask" \
    NUMEXPR_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# Ensure temp directories exist
RUN mkdir -p /data/tmp/pytensor /data/tmp/jax /data/tmp/dask && \
    chmod -R 777 /data/tmp

# Global PyTensor config
RUN printf "[global]\nfloatX = float64\ndevice = cpu\n\n[blas]\nldflags = -lopenblas\n" > /etc/pytensorrc

# Mark the workspace as safe for Git
RUN git config --system --add safe.directory /workspaces/conduit

USER appuser
WORKDIR /workspaces/conduit

CMD ["python3"]
