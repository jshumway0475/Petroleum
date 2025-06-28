# Use the latest GDAL-enabled base image from GitHub Container Registry
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# Set working directory
WORKDIR /app

# Install system dependencies and Python venv support
RUN apt-get update && \
    apt-get install -y \
        git curl unixodbc unixodbc-dev \
        python3 python3-pip python3-venv \
        postgresql-client sudo \
        libomp-dev vim wget g++ libopenblas-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up and activate a virtual environment
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip

# Make sure all future pip/python commands use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install them into the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (for VS Code and security best practices)
RUN useradd -m appuser && \
    echo "appuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/appuser && \
    chmod 0440 /etc/sudoers.d/appuser

# Switch to non-root user
USER appuser

# Set environment variables for Python behavior and parallel processing
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Optional: PyTensor config for PyMC
RUN echo "[global]\nfloatX = float64\ndevice = cpu\n\n[blas]\nldflags = -lopenblas\n" > /home/appuser/.pytensorrc

# Set workspace for mounted repo (used by devcontainer.json)
WORKDIR /workspace

# Default command for interactive sessions in VS Code
CMD ["python3"]
