# Usage Guide for PlayInsight Suite

This document explains how to run the PlayInsight Suite using Docker and Visual Studio Code Dev Containers.

---

## 🔧 Prerequisites

Make sure the following tools are installed on your system:

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Visual Studio Code](https://code.visualstudio.com/)
* [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu installed  
  - Recommended for Windows users who want to work in a Linux terminal
  - Ensure you've installed Ubuntu from the Microsoft Store and set it as your default WSL distro
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
* Create a local virtual environment at `.venv`
* Install the project in editable mode using `pip install --no-cache-dir -e .`
VS Code will detect and use the `.venv` environment automatically when you open the workspace.

---

## 🐳 Running the Docker Image Without VS Code (Optional)

If you're not using VS Code, you can pull and run the image directly:

```bash
docker pull jshumway0475/playinsight-image:latest
```

Then run it:

```bash
docker run -it --rm -v $(pwd):/workspace -w /workspace jshumway0475/playinsight-image python3
```
> 💡 If you're using PowerShell or CMD on Windows, replace `$(pwd)` with the full path to your project directory (e.g., `C:/Users/yourname/foldername`)

This launches a clean, fully configured environment with all required dependencies pre-installed.

**Note:** The container uses a non-root user (`appuser`) with passwordless sudo access. You will not be prompted for a password when installing additional packages or running privileged commands inside the container.

---

## 🤔 Running the Workflow

Once inside the container (via VS Code or Docker):

1. **Edit the config**
   Modify `config/analytics_config.yaml` with your database connection info and default parameters.

2. **Run workflows manually or on a schedule**
   Example:

   ```bash
   python play_assessment_tools/arps_autofit.py
   ```

3. **Populate your database**
   Use the scripts in the `sql/` folder to create tables, stored procedures, and views.

4. **Connect to Spotfire or another BI tool**
   Use the SQL views created to visualize spacing, parent-child relationships, forecasts, etc.

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
