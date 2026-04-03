# Getting started from scratch

This guide assumes you have **never** used this project before. You do not need to know machine learning theory to run the app; you only need to follow the steps and use your own spreadsheet-style data (CSV).

---

## 1. What this program does (in plain terms)

**Meta AI** is a program that runs **on your computer** in a **web browser**. You upload a table of data (usually a **CSV file**), tell the app which column you want to **predict**, and the software helps you clean the data, train models, show charts, and optionally **export** a package you can deploy elsewhere.

Nothing is sent to the cloud **unless** you configure API keys for optional features (for example certain AI assistants). The main workflow works **offline** on your machine after dependencies are installed.

---

## 2. What you need installed

| Item | Why |
|------|-----|
| **Python 3.10 or newer** | The application is written in Python. |
| A **web browser** (Chrome, Edge, Firefox, etc.) | The interface opens like a website at an address such as `http://127.0.0.1:7860`. |
| **Enough disk space** | Python packages can be large (several gigabytes total is common). |
| **RAM** | More data and more complex training need more memory. Start with a small CSV if unsure. |

If Python is not installed, download it from [python.org](https://www.python.org/downloads/) and use the installer. On Windows, enable the option **“Add Python to PATH”** during installation.

---

## 3. Get the project folder on your machine

You need the project files on disk. Typical options:

- **You already have a folder** (for example `meta_ai_builder_pp` on your Desktop). Use that path in all commands below.
- **You use Git**: open a terminal, `cd` to where you keep projects, then clone the repository and `cd` into the folder name you cloned.

Every command in this guide assumes the terminal’s **current directory** is the **root of the project** (the folder that contains `quick_start.py` and `requirements.txt`).

---

## 4. Use a virtual environment (recommended)

A **virtual environment** keeps this project’s libraries separate from other Python projects.

**Windows (Command Prompt or PowerShell):**

```text
cd path\to\meta_ai_builder_pp
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
cd /path/to/meta_ai_builder_pp
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal prompt often shows `(.venv)`. Keep this terminal open for the next steps.

**Alternative (Windows only):** run `scripts\setup.bat` once from the project folder. It creates useful subfolders and runs `pip install -r requirements.txt`. You still need Python installed first.

---

## 5. Install dependencies

With the virtual environment **activated** and your terminal **in the project root**:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run can take **many minutes** because it downloads and compiles scientific libraries. Wait until the command finishes without errors.

---

## 6. Start the application

Still in the project root, with the virtual environment active:

```bash
python quick_start.py
```

**Windows shortcut:** double-click `RUN_GRADIO_APP.bat` in the project folder (it tries to use `.venv312`, `.venv`, or system Python).

The terminal will print a line similar to:

```text
Access: http://127.0.0.1:7860
```

The port number might differ (for example `7861`) if something else is using the default port.

---

## 7. Open the interface in your browser

1. Leave the terminal **running**. Closing it or pressing Ctrl+C stops the app.
2. Open your browser and go to the **exact URL** printed in the terminal (for example `http://127.0.0.1:7860`).

If the page does not load, see [Troubleshooting](#10-troubleshooting) below.

---

## 8. Your first end-to-end path (minimal)

Use a CSV where **one column** is what you want to predict (for example “will churn”, “price”, “risk score”). Other columns are **inputs**.

1. Go to **Data Ingestion → Manual Upload** and upload your CSV.
2. When the app asks, choose the **target column** (the column to predict).
3. Explore other tabs when you are ready; they are optional on the first day.
4. Open **Model Training → Normal Training** and run training. Wait until it finishes.
5. Open **Analysis & XAI** to see results when available.
6. When you are satisfied, use **Export** to download a deployment package (after a successful train in the **same session**).

You can repeat uploads and training as you learn; start with **small files** so each step finishes quickly.

---

## 9. “Session” and “deployment” are different things

People often mix these up because both words appear in the docs.

**Session (dashboard session)**  
From the moment you run `python quick_start.py` until you stop that process (Ctrl+C or close the terminal). During that time the app remembers your upload and trained model in memory. **Export** usually requires that you already **trained** in that **same** run, without restarting the app.

**Deployment (hosting the project)**  
A separate topic: how to **run this codebase or an Export ZIP on a server or in Docker** so others can reach it. Step-by-step commands: **[HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md)**. The **MLOps and Production** *tabs* in the app are not the same thing; those are in-UI tools.

There is no official name **“deployment session.”** If you see **same session**, it always means **same Gradio run**, not “a Docker session” or “a cloud session.”

---

## 10. Names you might see

| Name | Meaning |
|------|--------|
| **Gradio** | The library that draws the web UI for this project. |
| **`quick_start.py`** | The script that starts the main dashboard. |
| **`dashboard_v3.py`** | The file that defines tabs and behavior (you usually do not edit it as a beginner). |
| **CSV** | Comma-separated values; a simple table format from Excel or other tools. |
| **Virtual environment** | An isolated Python library folder for this project. |

---

## 11. Troubleshooting

| Problem | What to try |
|--------|-------------|
| `python` is not recognized | Reinstall Python with “Add to PATH”, or use `py` on Windows instead of `python`. |
| `pip install` fails | Check internet connection; upgrade pip: `python -m pip install --upgrade pip`. |
| Browser says **connection refused** | The app is not running; run `python quick_start.py` again and use the printed URL. |
| Port already in use | Close other programs using that port, or set environment variable `GRADIO_SERVER_PORT` to another port before starting. |
| Training or export errors | Ensure you selected a target column and completed training in the **same** session before export. |

---

## 12. What to read next

| Document | When to use it |
|----------|----------------|
| [README.md](../README.md) | Full install options, all tabs, MLflow, troubleshooting. |
| [HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md) | How to **deploy / host** this project (servers, Docker, export API). |
| [PRD.md](PRD.md) | What the product officially includes and excludes (scope and non-goals). |

---

## 13. Getting help in the real world

- Save the **exact error message** from the terminal (copy and paste).
- Note your **Python version** (`python --version`) and **operating system**.
- Describe whether the problem happens at **install**, **start**, **upload**, **train**, or **export**.

That information is what any engineer needs to diagnose issues quickly.
