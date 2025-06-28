# Usage Guide for PlayInsight Suite

This document explains how to run the PlayInsight Suite using Docker and Visual Studio Code Dev Containers.

---

## 🔧 Prerequisites

Make sure the following tools are installed on your system:

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Visual Studio Code](https://code.visualstudio.com/)
* VS Code Extensions:

  * ✅ Remote - Containers (aka Dev Containers)
  * ✅ Python
  * ✅ Docker

---

## 🧪 Getting Started with Dev Containers (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/jshumway0475/Petroleum.git
cd Petroleum
```

### 2. Open the Folder in VS Code

Go to **File → Open Folder**, and select the `Petroleum` directory you just cloned.

### 3. Reopen in Dev Container

You should be prompted to **"Reopen in Container"**. Click Yes.

> If not prompted, press `Ctrl+Shift+P` and run:
> `Dev Containers: Reopen in Container`

VS Code will now:

* Build the Docker image
* Mount your code into the container
* Install the project in editable mode using `pip install -e .`

---

## 🐳 Using the Docker Image Directly

If you're not using VS Code, you can pull and run the image directly:

```bash
docker pull jshumway0475/playinsight-image:latest
```

Then run it:

```bash
docker run -it --rm -v ${PWD}:/workspace -w /workspace jshumway0475/playinsight-image python3
```

This drops you into a clean environment with all required dependencies.

---

## 🧐 Running the Workflow

Once inside the container (via VS Code or Docker):

1. **Edit the config**
   Modify `config/analytics_config.yaml` with your database connection info and default parameters.

2. **Run workflows manually or on a schedule**
   Example:

   ```bash
   python play_assessment_tools/arps_autofit.py
   ```

3. **Populate your database**
   Use the scripts in `sql/` to create tables, stored procedures, and views.

4. **Connect to Spotfire or another BI tool**
   Use the views created by the SQL scripts to visualize parent-child assignments, forecasts, spacing, etc.

---

## 🔄 Rebuilding the Container

If you change `requirements.txt` or `Dockerfile`, rebuild the container:

```bash
Ctrl+Shift+P → Dev Containers: Rebuild Container
```

---

## 🚘 Need Help?

If you encounter issues or need support, contact:

📧 [jshumway0475@gmail.com](mailto:jshumway0475@gmail.com)

---
