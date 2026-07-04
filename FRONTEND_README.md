# Web App — React (shadcn/ui) + FastAPI

Author: Aman Kushwah (2024AC05064) — Group 105

A working web application for the Predictive-Maintenance model: enter live hydraulic sensor
readings, get an instant verdict, confidence, explanation, and RAG-retrieved repair guidance.
This is the **"application" the assignment asks you to screenshot.**

- **Frontend:** React + Vite + TypeScript + Tailwind + **shadcn/ui** (`frontend/`)
- **Backend:** FastAPI (`api_server.py`) — serves the Random Forest model, guarded by the
  security layer, enriched by the RAG advisor.

Screenshots are in `docs/screenshots/` (`app_input.png`, `app_prediction.png`).

---

## Prerequisites
- Python 3.10+ (backend) and Node.js 18+ (frontend). Both confirmed working with the versions
  in this repo.

## Run it locally (two terminals)

**Terminal 1 — backend (port 8000):**
```bash
cd mediclaim-ai-insurance-agent

# one-time: create venv + install python deps
python3 -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements-notebook.txt fastapi uvicorn

# train + save the model once (creates model_registry/…joblib)
python3 train_and_save.py

# start the API
uvicorn api_server:app --reload --port 8000
#   docs at http://localhost:8000/docs
```

**Terminal 2 — frontend (port 5173):**
```bash
cd mediclaim-ai-insurance-agent/frontend
npm install          # one-time
npm run dev
#   open http://localhost:5173
```

Then open **http://localhost:5173**, click a preset (Healthy / Stressed) or type sensor values,
and press **Assess Machine**.

> Tip: open `http://localhost:5173/?demo=1` to auto-run one assessment on load (used for the
> screenshots).

## What the app demonstrates
- **ML inference** — Random Forest verdict + confidence.
- **Explainability** — top contributing sensors shown as chips.
- **RAG advisor** — retrieved repair procedure for the flagged component.
- **Security** — the API validates input ranges, checks the API key, rate-limits, and audits
  every request (out-of-range values return HTTP 400).

## How it maps to the architecture
- The **React app** is the non-ML presentation layer.
- The **FastAPI backend** is the real-time inference microservice.
- It reuses `src/security_layer.py` (Security Gateway) and `src/maintenance_advisor.py`
  (RAG Retriever) — the same components shown in the architecture diagram.

## Production build (optional)
```bash
cd frontend && npm run build      # outputs frontend/dist/
```
