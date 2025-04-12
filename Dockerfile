# Use the OSGEO GDAL image with Ubuntu as the build stage
FROM osgeo/gdal:ubuntu-small-3.6.3 AS builder

# Set working directory
WORKDIR /app

# Install system dependencies and Git
RUN apt-get update && \
    apt-get install -y git openssh-client curl unixodbc unixodbc-dev \
    python3 python3-pip postgresql-client sudo libomp-dev vim wget g++ \
    libopenblas-dev && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip

# Set the ARG for the GitHub PAT
ARG GITHUB_TOKEN

# Clone the repository
RUN git clone https://${GITHUB_TOKEN}@github.com/Jay-Engineering/conduit.git /app/conduit

# Set working directory to the cloned repository
WORKDIR /app/conduit

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Install the repository using pip
RUN pip3 install --no-cache-dir .

# Install Java (ensure we can locate openjdk-17-jdk)
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean

# Set JAVA_HOME variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Create a non-root user
RUN useradd -m vscode && \
    echo "vscode ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/vscode && \
    chmod 0440 /etc/sudoers.d/vscode

# Set the user to the non-root user
USER vscode

# Expose any ports the app is expecting in the environment
EXPOSE 80

# Set environment variables for Python and Pip
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/vscode/.local/bin:$PATH

# Set additional environment variables
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Set PyTensor configuration through a configuration file
RUN echo "[global]\n" \
         "floatX = float64\n" \
         "device = cpu\n" \
         "\n" \
         "[blas]\n" \
         "ldflags = -lopenblas\n" > /home/vscode/.pytensorrc

# Set the default command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
