# Usage Guide for PlayInsight Suite

This document explains how to run the PlayInsight Suite using Docker and Visual Studio Code Dev Containers.

---

## 🍴 Fork & Clone

1. On GitHub, click **Fork** (upper-right) to create your own copy under your account.

2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/playinsight.git
   cd playinsight
   ```

3. Add the upstream remote so you can pull in future fixes:

   ```bash
   git remote add upstream https://github.com/jshumway0475/Petroleum.git
   git fetch upstream
   ```

---

## 🔧 Prerequisites

Make sure the following tools are installed on your system:

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Visual Studio Code](https://code.visualstudio.com/)
* [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu installed

  * Recommended for Windows users who want to work in a Linux terminal
  * Ensure you've installed Ubuntu from the Microsoft Store and set it as your default WSL distro
* VS Code Extensions:

  * Remote - Containers (aka Dev Containers)
  * Python
  * Docker

---

## 🧪 Getting Started with Dev Containers (Recommended)

### 1. Set Up a Local Projects Folder and Clone the Repository

Choose or create a local folder where you keep code projects.

**If using WSL:**

```bash
cd /mnt/c/projects     # or your preferred folder
git clone https://github.com/<your-username>/playinsight.git
cd playinsight
```

**If using Git in Windows PowerShell or CMD:**

```powershell
cd C:\projects
git clone https://github.com/<your-username>/playinsight.git
cd playinsight
```

### 2. Set Local Projects Path Environment Variable

Set an environment variable on your host machine pointing to your full path:

#### On PowerShell:

```powershell
$env:LOCAL_PROJECTS_PATH = "C:\Users\yourname\your-folder"
```

#### On WSL/Linux:

```bash
export LOCAL_PROJECTS_PATH="/mnt/c/Users/yourname/your-folder"
```

#### Optional: Add to your shell profile to persist it:

```bash
echo 'export LOCAL_PROJECTS_PATH="/mnt/c/Users/yourname/your-folder"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Open the Folder in VS Code

In VS Code, go to File → Open Folder, and select the `playinsight` directory you just cloned.

### 4. Reopen in Dev Container

You should be prompted to "Reopen in Container". Click Yes.

If not prompted, press `Ctrl+Shift+P` and run:

```
Dev Containers: Reopen in Container
```

VS Code will now:

* Build the Docker image
* Mount your code into the container
* Create a local virtual environment at `.venv`
* Install the project in editable mode using:

```bash
pip install --no-cache-dir -e .
```

VS Code will automatically detect and use the `.venv` environment.

---

## 🔄 Syncing with Upstream

Whenever you want the latest from the main repo:

```bash
git fetch upstream
git checkout main
git merge upstream/main
# resolve conflicts if any, then:
git push origin main
```

---

## 🛠 Optional: Add a `devcontainer.local.json` for Personal Customization

To avoid editing the shared `devcontainer.json` (which is version-controlled), you can create a `devcontainer.local.json` file in the `.devcontainer/` folder. This allows you to override settings like mounts or environment variables without affecting other users.

This approach is especially useful when you want to bind your own local development folder to the container’s `/workspace` directory.

---

### ✅ Why Use `devcontainer.local.json`?

* Keeps machine-specific configuration out of version control
* Lets each user define their own mount paths or environment settings
* VS Code automatically merges it with `devcontainer.json`, giving priority to local values

---

### 📁 Example: `.devcontainer/devcontainer.local.json`

```json
{
  "mounts": [
    "source=/mnt/c/projects,target=/workspace,type=bind"
  ]
}
```

💡 Replace `/mnt/c/projects` with the full path to your local development folder.
For example, on Windows using WSL: `/mnt/c/Users/yourname/your-folder`.

---

## 🔐 Per-Machine Config Overrides

Drop any sensitive or machine-specific settings into a file that’s already in `.gitignore`, for example:

```yaml
# config/analytics_config.local.yaml
 db_host: mydb.example.com
 db_user: alice
 db_pass: secret
```

Your scripts or loaders will look for `analytics_config.local.yaml` first, then fall back to the checked-in defaults.

---

## 🐳 Optional: Run the Docker Image Without VS Code

If you're not using VS Code, you can pull and run the image directly:

```bash
docker pull jshumway0475/playinsight-image:latest
```

Then run it:

```bash
docker run -it --rm -v $(pwd):/workspace -w /workspace jshumway0475/playinsight-image python3
```

💡 If you're using PowerShell or CMD on Windows, replace `$(pwd)` with the full path to your project directory, like:

```powershell
-v C:/Users/yourname/projects/playinsight:/workspace
```

This launches a clean, fully configured environment with all required dependencies pre-installed.

Note: The container uses a non-root user (`appuser`) with passwordless sudo access. You will not be prompted for a password when installing additional packages or running privileged commands inside the container.

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

If you change `requirements.txt` or `Dockerfile`, rebuild the container by running:

```bash
Ctrl+Shift+P → Dev Containers: Rebuild Container
```

---

## 🚘 Need Help?

If you encounter issues or need support, contact:

📧 [jshumway0475@gmail.com](mailto:jshumway0475@gmail.com)
