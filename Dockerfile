# Use latest GDAL-enabled image with Ubuntu
FROM osgeo/gdal:ubuntu-small-latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git curl unixodbc unixodbc-dev \
        python3 python3-pip postgresql-client sudo \
        libomp-dev vim wget g++ libopenblas-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy and install requirements (expect these to be in context)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser && \
    echo "appuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/appuser && \
    chmod 0440 /etc/sudoers.d/appuser

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Optional: Configure PyTensor for PyMC
RUN echo "[global]\nfloatX = float64\ndevice = cpu\n\n[blas]\nldflags = -lopenblas\n" > /home/appuser/.pytensorrc

# === Your local repo will be mounted at /workspace via VS Code ===
WORKDIR /workspace

# Install the mounted Petroleum repo in editable mode
RUN pip3 install --no-cache-dir -e .

# Default command — launch an interactive Python session
CMD ["python3"]
