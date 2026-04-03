# How to deploy this project

This page explains **how to run Meta AI outside your personal laptop session**: on a server, in Docker, or as an inference-only API. It is **not** documentation for the **MLOps and Production** area inside the Gradio app (that is product UI for readiness checks, snippets, and MLflow status).

You deploy **the codebase** you cloned from this repository (or an image built from it). Training in the browser is unchanged; deployment only answers: *where does `quick_start.py` or the exported API run, and who can reach it?*

**Prerequisites:** Python 3.10+, dependencies installed (`pip install -r requirements.txt`), project root containing `quick_start.py`. For Docker, install Docker Engine and Docker Compose.

---

## Choose a deployment shape

| Goal | What you run | Typical use |
|------|----------------|-------------|
| **A. Full dashboard on a network** | `quick_start.py` bound to `0.0.0.0` | Team uses the same training UI from browsers over VPN or LAN |
| **B. Docker (one container)** | Image from `Dockerfile.prod` | Repeatable installs, same behavior on any Linux host |
| **C. Docker Compose** | `docker-compose.metaai.yml` | Gradio on 7860 **and** FastAPI (`backend_api_main`) on 8000 together |
| **D. Inference only** | `uvicorn api:app` on an **Export** ZIP | Production scoring API without the training UI |

---

## A. Full dashboard on a server or VM

On the machine that will host the app:

1. Clone or copy the project and install dependencies (see [GETTING_STARTED.md](GETTING_STARTED.md)).
2. Bind Gradio to all interfaces so remote browsers can connect:

```bash
cd /path/to/meta_ai_builder_pp
source .venv/bin/activate   # or Windows: .venv\Scripts\activate

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
python quick_start.py
```

3. Open `http://<server-ip>:7860` from another machine (firewall must allow the port).
4. For HTTPS and passwords, put **nginx**, **Caddy**, or your cloud load balancer **in front** of that port. Do not expose plain HTTP on the public internet without TLS.

---

## B. Docker: single container (Gradio only)

From the **project root** (where `Dockerfile.prod` lives):

```bash
docker build -f Dockerfile.prod -t meta-ai:local .
docker run --rm -p 7860:7860 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7860 \
  -v "$(pwd)/data:/app/data" \
  meta-ai:local
```

Windows PowerShell: use `-v "${PWD}/data:/app/data"` or an absolute path instead of `$(pwd)`.

Browse to `http://<host>:7860`. Add volume mounts for `exports/` or `models/` if you persist artifacts on disk.

---

## C. Docker Compose (Gradio + FastAPI)

Starts the **training UI** and the repository’s **FastAPI** app together:

```bash
docker compose -f docker-compose.metaai.yml up --build
```

- **Gradio (dashboard):** `http://localhost:7860` — `python quick_start.py`
- **FastAPI (`backend_api_main`):** `http://localhost:8000` — health: `/health`, docs: `/api/docs`

The API service runs `uvicorn backend_api_main:app`. It is **not** the small bundle produced by the in-app **Export** tab; that bundle uses `api:app` from the ZIP (see D).

---

## D. Inference-only (recommended for production scoring)

1. Use the **running dashboard** to train a model.
2. In the **Export** tab, download the ZIP (same Gradio run as training).
3. On a clean server, unzip, install `requirements.txt` from the bundle, then:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Point clients at `http://<host>:8000` (and `/docs` if enabled). Scale this service behind a load balancer; it is **stateless** once running.

---

## Kubernetes

Use `kubernetes_deployment.yaml` as a **template**. Replace image names, secrets, resource limits, and ingress with values for your cluster and registry. Align health probes with the HTTP routes your image actually serves (Gradio vs FastAPI).

---

## Production checklist

| Topic | Action |
|-------|--------|
| TLS | Terminate HTTPS at the load balancer or ingress. |
| Auth | Do not rely on Gradio for multi-user security; use SSO or API gateway in front. |
| Secrets | Store API keys in env vars or a secret manager. |
| Data | Mount persistent disks for `data/`, `exports/`, MLflow DB if you keep them. |
| Training UI | Usually low concurrency; scale **inference** (D) for heavy traffic. |

---

## Where this fits in the docs

| Document | Content |
|----------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | First install and first CSV run on your PC |
| **This file** | Host the project or the export on a network |
| [README.md](../README.md) | Full reference, all tabs, troubleshooting |

---

*Internal UI note: **MLOps and Production** tabs inside the app help you with readiness text, API snippets, drift views, and MLflow. They do not replace the steps above for actually hosting the software.*
