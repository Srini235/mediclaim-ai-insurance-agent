# Contribution Notes — Aman Kushwah (2024AC05064), Group 105

This file documents the part of Assignment I that I built: the **ML model, the RAG
maintenance advisor, the application-wide security layer, and the test suite** for the
*Predictive Maintenance of Mobile Hydraulic Systems* solution. It's meant to help the team
understand exactly what I added and how to run/verify it.

---

## 1. What I added (files)

| File | What it is |
|:--|:--|
| `105.ipynb` | Self-contained solution notebook — data pipeline, model, quality-requirement demos, patterns, security, RAG. Fully executed with outputs. |
| `105.html` | Rendered HTML export of the executed notebook (open in any browser, no setup needed). |
| `src/security_layer.py` | Reusable, application-wide security controls (used by the notebook **and** the FastAPI services). |
| `src/maintenance_advisor.py` | RAG "Maintenance Advisor" — retrieves the right repair procedure when the model flags a machine. |
| `data/hydraulic_maintenance_manual.md` | Knowledge base of maintenance procedures used by the RAG retriever. |
| `tests/test_predictive_maintenance.py` | 18 pytest tests covering the model, security, and RAG. |
| `build_notebook.py` | Script that generates `105.ipynb` (so the notebook is reproducible, not hand-edited). |
| `model_registry/` | Versioned model artifacts written by the Model Registry pattern. |

> Note on scope: the team pivoted from the earlier *health-insurance* idea to
> *predictive maintenance of mobile hydraulics*. This notebook is the new solution. The old
> `src/agents`, `src/core`, `docker_compose.yml` etc. are from the insurance MVP and are not
> part of this deliverable.

---

## 2. What the solution does

A construction fleet services excavator **hydraulic systems** on a fixed schedule. That wastes
money on healthy parts and misses parts that fail early. We put sensors on the machine (pressure,
temperature, vibration, flow, oil debris) and use ML to service parts **only when needed** — a
smarter "check-engine light" that says *which* part, *how confident*, and *why*.

**Flow:** sensor data → Pipe-and-Filter cleaning → Random Forest prediction → (if flagged) RAG
retrieves the repair procedure → served via batch or real-time path, all behind the security layer.

---

## 3. How the assignment requirements are covered

**Objective 1 — Requirements (GR4ML)**
- Business View, Analytics Design View, Data Preparation View, and the top-3 quality requirements
  are written up in the notebook (sections 1–5) and demonstrated in code.

**Objective 2 — Architecture & Patterns**
- **Architectural patterns:** Pipe-and-Filter (data pipeline) + Microservices (decoupled services).
- **Design patterns:** Model Registry (versioned artifacts) + Batch-vs-Real-time Serving.
- Both are implemented and run in the notebook.

**Quality requirements (each measured in the notebook):**
1. **Robustness** — 15% of sensor data corrupted, accuracy retained ≈ 99% (PASS).
2. **Latency** — real-time inference ≈ 50 ms, target < 100 ms (PASS).
3. **Explainability** — feature importances + per-machine top factors.

**Extras I added**
- **Security (application-wide):** input validation, API-key auth, model integrity (SHA-256),
  tamper-evident audit log, rate limiting — all demonstrated in the notebook and unit-tested.
- **RAG Maintenance Advisor:** semantic retrieval of repair procedures. Uses TF-IDF + cosine
  similarity for a zero-dependency prototype; **ChromaDB is the production vector store**
  (same pattern as the Session 5 lab demo).

---

## 4. How to run it on your local machine

**Prerequisites:** Python 3.10+ and the packages below (all lightweight — no API keys, no GPU,
no internet needed at run time).

```bash
# 1. from the repo root, install dependencies
pip install pandas numpy scikit-learn matplotlib joblib jupyter nbconvert pytest

# 2a. OPEN and run interactively (best for taking screenshots)
jupyter notebook 105.ipynb
#    then: Kernel -> Restart & Run All

# 2b. OR just view the already-executed result — open 105.html in a browser
```

**Run the notebook headless (no UI) and regenerate outputs:**
```bash
python3 -c "import nbformat; from nbconvert.preprocessors import ExecutePreprocessor as E; nb=nbformat.read('105.ipynb',as_version=4); E(timeout=300,kernel_name='python3').preprocess(nb,{'metadata':{'path':'.'}}); nbformat.write(nb,'105.ipynb'); print('ALL CELLS PASSED')"
```
If it prints `ALL CELLS PASSED`, every cell ran with no error.

**Regenerate the notebook from scratch (optional):**
```bash
python3 build_notebook.py     # writes a fresh 105.ipynb, then run it as above
```

---

## 5. How to test it

**Run the full automated test suite (18 tests):**
```bash
# from the repo root
python3 -m pytest tests/test_predictive_maintenance.py -q
```
Expected: `18 passed`. These cover the model (accuracy, robustness, explainability), all five
security controls, and the RAG retriever.

**Manually verify the security controls (prove they block bad input):**
```bash
python3 -c "
from src.security_layer import validate_sensor_payload, ApiKeyAuthenticator, SecurityError
try: validate_sensor_payload({'pressure':99999,'temperature':70,'vibration':4,'flow_rate':85,'oil_debris':60})
except SecurityError as e: print('BLOCKED bad input ->', e)
try: ApiKeyAuthenticator(['good']).authenticate('wrong')
except SecurityError as e: print('BLOCKED bad key ->', e)
"
```

**Manually verify the RAG advisor:**
```bash
python3 -c "
from src.maintenance_advisor import MaintenanceAdvisor
print(MaintenanceAdvisor().advise('pump_leakage', k=1)[0]['procedure'])
"
```
Expected: `Internal pump leakage`.

---

## 6. Notes for the report / documentation
- The notebook prints clear, screenshot-ready output for each quality requirement and pattern.
- Security can be listed as a 4th quality requirement (or swapped into the top-3) — it's fully
  implemented, not just described.
- For the RAG section, state: *"ChromaDB in production (per Session 5 lab), TF-IDF retriever in this
  prototype for zero-dependency reproducibility."* This is accurate and course-aligned.
- Everything runs offline — no OpenAI key required — which makes it trivially reproducible for
  evaluation.
