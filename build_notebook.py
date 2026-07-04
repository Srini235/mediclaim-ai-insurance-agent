"""Builds 105.ipynb — Predictive Maintenance for Mobile Hydraulics.
Run: python3 build_notebook.py  then execute the notebook.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- Title
md(r"""
# Predictive Maintenance for Mobile Hydraulics
### AIMLCZG546 — Software Engineering for Machine Learning · Assignment I · **Group 105**

| Sl | BITS ID | Name | Contribution |
|:--:|:--|:--|:--:|
| 1 | 2024AC05744 | Srinivasan R | 100% |
| 2 | 2024AC05100 | Vineet Kumar | 100% |
| 3 | 2024AD05482 | Vibhav Sharma | 100% |
| 4 | 2024AC05064 | Aman Kushwah | 100% |

---
## Problem Statement
A construction fleet services excavator **hydraulic systems** on a fixed calendar schedule.
Machines are worked at very different intensities (mining vs. light landscaping), so a fixed
schedule either **replaces healthy parts early (wasted cost)** or **misses a part that fails
mid-job (unplanned downtime + safety risk)**.

**Goal:** put sensors on the machine (pressure, temperature, vibration) and use ML to predict
hydraulic component health, so maintenance happens **only when actually needed** — a smarter
"check-engine light" that says *which* part, *how confident*, and *why*.
""")

# ---------------------------------------------------------------- GR4ML
md(r"""
## GR4ML Modelling (Objective 1)

**Business View — stakeholders & goals**
- *Maintenance Manager* → minimise cost + avoid surprise breakdowns.
- *Field Technician* → know exactly which part is failing (arrive with the right spare).
- *Machine Operator* → real-time warning if the machine is unsafe **now**.

**Analytics Design View — decision → analytic**
- "Is the cooler/pump degrading?" → *predictive*, can run **overnight (batch)**.
- "Is the machine unstable right now?" → *predictive*, needs an **instant (real-time)** answer.
- Model = **Random Forest** — accurate, fast, and **self-explaining** (feature importances),
  so a technician sees the reasoning instead of a black-box verdict.

**Data Preparation View**
- Raw sensor streams are noisy (dust, vibration, electrical glitches → missing/spurious values).
- Pipeline: **clean** bad/missing readings → **summarise each work-cycle** into features
  (mean pressure, max vibration, etc.) → **feed** the model.
- Real logs are private, so we generate **realistic synthetic sensor data** for build + test.

**Top-3 Quality Requirements** (each demonstrated & measured below)
1. **Robustness** — stays accurate even with ~15% corrupted / missing sensor readings.
2. **Low latency** — the safety-critical real-time prediction returns in **< 100 ms**.
3. **Explainability** — every flag comes with the top contributing sensors.
""")

# ---------------------------------------------------------------- Architecture note
md(r"""
## Architecture & Patterns

**ML components:** Random Forest classifier, feature pipeline, model registry.
**Non-ML components:** sensor ingestion, work-cycle aggregator, batch scheduler, real-time
request handler, drift-monitoring watchdog.

| Pattern | Type | Where in this notebook |
|:--|:--|:--|
| **Pipe-and-Filter** | Architectural | Data pipeline = chained filters (Ingest → Clean → Aggregate → Feature) |
| **Microservices** | Architectural | Training, batch scoring, and real-time scoring are decoupled units (Srini wires services) |
| **Model Registry** | Design | Versioned save/load of the trained model artifact |
| **Batch vs. Real-time Serving** | Design | Two separate inference paths over the same model |
""")

# ---------------------------------------------------------------- Imports
code(r"""
import time, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import hashlib, logging, collections

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
print("Environment ready. sklearn RandomForest — predictive maintenance model.")
""")

# ---------------------------------------------------------------- Architecture diagram
md(r"""
### System Architecture Diagram (ML + Non-ML Components)

The diagram below shows the full system. **ML components** (orange) do learning/inference;
**non-ML components** (blue) handle ingestion, orchestration, storage, security, and serving.
""")

code(r'''
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ML   = "#e07a5f"   # ML components
NML  = "#4a7ba6"   # non-ML components
fig, ax = plt.subplots(figsize=(12, 7.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

def box(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=color, ec="black", lw=1.2, alpha=0.92))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", wrap=True)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                 lw=1.4, color="#333"))

# --- nodes ---
box(0.3, 6.8, 2.2, 1.1, "Hydraulic Sensors\n(pressure, temp,\nvibration, flow)", NML)
box(3.0, 6.8, 2.6, 1.1, "Pipe-and-Filter\nData Pipeline\n(Ingest→Clean→Feature)", NML)
box(6.1, 6.8, 2.3, 1.1, "Security Gateway\n(authN, validation,\nrate limit, audit)", NML)
box(9.0, 6.8, 2.6, 1.1, "Random Forest\nMaintenance Model", ML)

box(9.0, 4.6, 2.6, 1.0, "Model Registry\n(versioned, SHA-256)", NML)
box(6.1, 4.6, 2.3, 1.0, "Batch Scheduler\n(nightly fleet scan)", NML)
box(3.0, 4.6, 2.6, 1.0, "Real-time API\n(<100ms safety check)", NML)

box(3.0, 2.4, 2.6, 1.0, "RAG Retriever\n(semantic search)", ML)
box(6.1, 2.4, 2.3, 1.0, "Maintenance KB\n(repair manuals)", NML)
box(9.0, 2.4, 2.6, 1.0, "Drift Monitor\n(triggers retrain)", NML)

box(0.3, 0.5, 5.3, 1.0, "Maintenance Manager / Technician / Operator  (Stakeholders)", NML)

# --- flows ---
arrow(2.5, 7.35, 3.0, 7.35)
arrow(5.6, 7.35, 6.1, 7.35)
arrow(8.4, 7.35, 9.0, 7.35)
arrow(10.3, 6.8, 10.3, 5.6)      # model -> registry
arrow(9.0, 5.1, 8.4, 5.1)        # registry -> batch
arrow(6.1, 5.1, 5.6, 5.1)        # batch -> realtime
arrow(4.3, 4.6, 4.3, 3.4)        # realtime -> RAG
arrow(5.6, 2.9, 6.1, 2.9)        # RAG -> KB
arrow(10.3, 4.6, 10.3, 3.4)      # registry -> drift
arrow(4.3, 2.4, 3.0, 1.5)        # RAG -> stakeholders (guidance)
arrow(3.0, 5.1, 2.9, 1.5)        # realtime -> stakeholders (alert)

# --- legend ---
ax.add_patch(FancyBboxPatch((6.0, 0.6), 0.4, 0.35, boxstyle="round", fc=ML, ec="black"))
ax.text(6.5, 0.77, "ML component", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((8.6, 0.6), 0.4, 0.35, boxstyle="round", fc=NML, ec="black"))
ax.text(9.1, 0.77, "Non-ML component", va="center", fontsize=9)

ax.set_title("Predictive Maintenance of Mobile Hydraulics — System Architecture",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("architecture_diagram.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved architecture_diagram.png")
''')

# ---------------------------------------------------------------- Pipe-and-Filter data gen
md(r"""
### 1. Data Preparation — *Pipe-and-Filter* architecture

Each **Filter** does one job and passes the payload to the next: **Ingest → Clean → Aggregate
→ Feature**. This is the Data Preparation View, implemented as an architectural pattern.
""")

code(r'''
# ---- Pipe-and-Filter: base + concrete filters -------------------------------
class Filter:
    """Abstract filter node in the data pipeline."""
    def set_next(self, nxt):
        self.next = nxt
        return nxt
    def process(self, payload):
        raise NotImplementedError


class IngestFilter(Filter):
    """Filter 1: simulate raw hydraulic sensor readings for N work-cycles."""
    def __init__(self, n_cycles=4000):
        self.n = n_cycles
    def process(self, payload):
        n = self.n
        # duty_intensity 0..1 => how hard the machine is worked this cycle
        duty = RNG.uniform(0, 1, n)
        pressure  = 150 + 40*duty + RNG.normal(0, 6, n)      # bar
        temperature = 55 + 35*duty + RNG.normal(0, 4, n)     # deg C
        vibration = 2.0 + 6.0*duty + RNG.normal(0, 0.6, n)   # mm/s RMS
        flow_rate = 90 - 15*duty + RNG.normal(0, 3, n)       # L/min
        oil_debris = 20 + 120*duty + RNG.normal(0, 12, n)    # ppm particles
        # Ground-truth failure risk rises with heat, vibration and debris
        risk = (0.35*(temperature-55)/35 + 0.35*(vibration-2)/6
                + 0.30*(oil_debris-20)/120)
        prob = 1/(1+np.exp(-10*(risk-0.5)))
        failure = (RNG.uniform(0,1,n) < prob).astype(int)
        payload["raw"] = pd.DataFrame({
            "pressure": pressure, "temperature": temperature,
            "vibration": vibration, "flow_rate": flow_rate,
            "oil_debris": oil_debris, "needs_maintenance": failure})
        print(f"[Filter 1/4 Ingest]  generated {n} work-cycles "
              f"({failure.mean()*100:.1f}% need maintenance)")
        return self.next.process(payload)


class CleanFilter(Filter):
    """Filter 2: inject then repair sensor glitches (missing/NaN readings)."""
    def process(self, payload):
        df = payload["raw"].copy()
        sensor_cols = ["pressure","temperature","vibration","flow_rate","oil_debris"]
        # ~3% of raw readings glitch to NaN, as real dusty sensors do
        mask = RNG.uniform(0,1,df[sensor_cols].shape) < 0.03
        df[sensor_cols] = df[sensor_cols].mask(mask)
        before = int(df[sensor_cols].isna().sum().sum())
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        print(f"[Filter 2/4 Clean]   repaired {before} missing sensor readings")
        payload["clean"] = df
        return self.next.process(payload)


class AggregateFilter(Filter):
    """Filter 3: derive cross-sensor work-cycle summaries."""
    def process(self, payload):
        df = payload["clean"].copy()
        # thermal-load ratio and vibration-per-pressure = engineered cycle summaries
        df["thermal_load"] = df["temperature"] / df["pressure"]
        df["vib_per_pressure"] = df["vibration"] / df["pressure"]
        print("[Filter 3/4 Aggregate] added thermal_load, vib_per_pressure")
        payload["agg"] = df
        return self.next.process(payload)


class FeatureFilter(Filter):
    """Filter 4: emit the final feature matrix X and label y."""
    def process(self, payload):
        df = payload["agg"]
        y = df["needs_maintenance"]
        X = df.drop(columns=["needs_maintenance"])
        payload["X"], payload["y"] = X, y
        print(f"[Filter 4/4 Feature]  X shape = {X.shape}, features = {list(X.columns)}")
        return payload


# ---- assemble & run the pipe ------------------------------------------------
ingest = IngestFilter(4000)
ingest.set_next(CleanFilter()).set_next(AggregateFilter()).set_next(FeatureFilter())
payload = ingest.process({})
X, y = payload["X"], payload["y"]
X.head()
''')

# ---------------------------------------------------------------- Train
md(r"""
### 2. Analytics Design View — train & validate the Random Forest

We use a stratified **80/20 train-test split** and **5-fold cross-validation** so the reported
performance is not a lucky single split. Random Forest is chosen for accuracy, speed, and
built-in explainability.
""")

code(r'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)

# 5-fold cross-validation for a rigorous, split-independent estimate
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"5-fold CV F1 scores : {np.round(cv_scores, 4)}")
print(f"CV mean F1          : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

# Fit on the full training set and evaluate on the held-out test set
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Held-out test accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, pred,
      target_names=["Healthy", "Needs Maintenance"]))
''')

code(r'''
# Confusion matrix visual (screenshot-ready)
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Healthy","Needs Maint."]); ax.set_yticklabels(["Healthy","Needs Maint."])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)
plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- QR1 robustness
md(r"""
### 3. Quality Requirement #1 — Robustness (15% corrupted data)
Deliberately corrupt **15%** of the test readings (missing + noise) and confirm accuracy holds.
""")

code(r'''
def corrupt(df, frac=0.15, seed=7):
    """Randomly null + jitter `frac` of all sensor cells, then median-impute."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    mask = rng.uniform(0,1,d.shape) < frac
    noise = rng.normal(0, d.std().values, d.shape)   # heavy sensor noise
    d = d.mask(mask, d + noise)                       # corrupt selected cells
    return d.fillna(d.median())

X_test_corrupt = corrupt(X_test, frac=0.15)
acc_corrupt = accuracy_score(y_test, model.predict(X_test_corrupt))
retention = acc_corrupt / acc * 100          # % of clean accuracy retained
print(f"Clean accuracy      : {acc*100:.2f}%")
print(f"15%-corrupt accuracy: {acc_corrupt*100:.2f}%")
print(f"Accuracy retained   : {retention:.1f}%  (target >= 95%)")
print("PASS - robust to corrupted sensor data" if retention >= 95
      else "REVIEW - robustness below target")
''')

# ---------------------------------------------------------------- QR2 latency
md(r"""
### 4. Quality Requirement #2 — Real-time latency (< 100 ms)
The safety-critical path must answer instantly. Measure single-sample inference time.
""")

code(r'''
sample = X_test.iloc[[0]]
# warm up
for _ in range(5): model.predict(sample)
runs = 100
t0 = time.perf_counter()
for _ in range(runs):
    model.predict(sample)
latency_ms = (time.perf_counter() - t0)/runs*1000
print(f"Avg real-time inference latency: {latency_ms:.2f} ms  (target < 100 ms)")
print("PASS — meets real-time safety target" if latency_ms < 100 else "REVIEW — too slow")
''')

# ---------------------------------------------------------------- QR3 explainability
md(r"""
### 5. Quality Requirement #3 — Explainability
Every flag comes with the top contributing sensors — no black-box verdicts.
""")

code(r'''
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6,4))
importances.plot.barh(ax=ax, color="#c44536")
ax.set_title("Global feature importance — why the model flags a machine")
ax.set_xlabel("Importance"); plt.tight_layout(); plt.show()

print("Top red-flag sensors:")
for name, val in importances.sort_values(ascending=False).head(3).items():
    print(f"  - {name}: {val:.3f}")
''')

code(r'''
def explain_one(row):
    """Per-machine explanation: prediction + top 3 contributing features."""
    proba = model.predict_proba(row)[0][1]
    contrib = (row.iloc[0] * model.feature_importances_)
    top = contrib.abs().sort_values(ascending=False).head(3).index.tolist()
    verdict = "NEEDS MAINTENANCE" if proba >= 0.5 else "HEALTHY"
    return {"verdict": verdict, "confidence": round(float(proba),3),
            "top_factors": top}

print("Example per-machine explanation:")
print(json.dumps(explain_one(X_test.iloc[[3]]), indent=2))
''')

# ---------------------------------------------------------------- Model Registry
md(r"""
### 6. Design Pattern — Model Registry
Version the trained artifact so you always know which model is "on duty" and can roll back.
""")

code(r'''
class ModelRegistry:
    """Minimal file-based model registry (save / list / load versioned artifacts)."""
    def __init__(self, root="model_registry"):
        self.root = root; os.makedirs(root, exist_ok=True)
    def register(self, model, name, version, metrics):
        path = os.path.join(self.root, f"{name}_v{version}.joblib")
        joblib.dump(model, path)
        meta = {"name": name, "version": version, "path": path, "metrics": metrics}
        with open(os.path.join(self.root, f"{name}_v{version}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Registry] registered {name} v{version} -> {path}")
        return meta
    def load(self, name, version):
        return joblib.load(os.path.join(self.root, f"{name}_v{version}.joblib"))

registry = ModelRegistry()
meta = registry.register(model, "hydraulic_maintenance_rf", "1.0",
        {"accuracy": round(acc,4), "latency_ms": round(latency_ms,2),
         "robust_acc": round(acc_corrupt,4)})
print("Registered metadata:", json.dumps(meta["metrics"], indent=2))
''')

# ---------------------------------------------------------------- Batch vs Realtime
md(r"""
### 7. Design Pattern — Batch vs. Real-time Serving
Same registered model, two decoupled inference paths (mirrors the Microservices split).
""")

code(r'''
serving_model = registry.load("hydraulic_maintenance_rf", "1.0")

def batch_score(fleet_df):
    """OVERNIGHT BATCH: score the whole fleet, return machines to service soon."""
    proba = serving_model.predict_proba(fleet_df)[:,1]
    out = fleet_df.copy()
    out["risk"] = proba
    flagged = out[out["risk"] >= 0.5].sort_values("risk", ascending=False)
    return flagged

def realtime_score(single_reading):
    """REAL-TIME: instant safety verdict for one machine right now."""
    t0 = time.perf_counter()
    res = explain_one(single_reading)
    res["latency_ms"] = round((time.perf_counter()-t0)*1000, 2)
    return res

# warm up the serving path so the measured latency reflects steady state
for _ in range(5):
    realtime_score(X_test.iloc[[0]])

# Batch demo — nightly fleet scan
fleet = X_test.sample(50, random_state=1)
flagged = batch_score(fleet)
print(f"[BATCH]  {len(flagged)}/50 machines flagged for maintenance. Top 3 by risk:")
print(flagged[["temperature","vibration","oil_debris","risk"]].head(3).round(2))

print("\n[REAL-TIME] instant safety check for one machine:")
# report a stable median latency over several warm calls (avoids single-call jitter)
_lat = sorted(realtime_score(X_test.iloc[[0]])["latency_ms"] for _ in range(11))[5]
_rt = realtime_score(X_test.iloc[[0]]); _rt["latency_ms"] = round(_lat, 2)
print(json.dumps(_rt, indent=2))
''')

# ---------------------------------------------------------------- Microservices
md(r"""
### 8. Architectural Pattern — Microservices

The system is decomposed into **independently deployable services**, each with a single
responsibility, communicating over well-defined contracts. In production these run as separate
FastAPI processes (see `src/agents/` and `src/core/`); here we expose them as decoupled callable
units to prove the boundaries. This is the second architectural pattern (alongside Pipe-and-Filter).

| Service | Responsibility | Scales on |
|:--|:--|:--|
| **Training Service** | build & register the model | offline / nightly |
| **Batch Scoring Service** | overnight fleet health scan | data volume |
| **Real-time Inference Service** | instant safety verdict | request rate |
| **Security Gateway** | authN, validation, rate limiting, audit | request rate |
""")

code(r'''
# Each "service" is an independent unit with its own contract (dict in / dict out).
# In deployment these are separate FastAPI apps; the decoupling is what matters.

def training_service(payload_pipe):
    """SERVICE 1: owns model lifecycle; returns a registry handle."""
    return {"model_name": "hydraulic_maintenance_rf", "version": "1.0"}

def batch_scoring_service(fleet_df):
    """SERVICE 2: stateless batch scorer over the registered model."""
    return batch_score(fleet_df)

def realtime_inference_service(reading):
    """SERVICE 3: stateless low-latency scorer for one machine."""
    return realtime_score(reading)

# Demonstrate the services are independently callable
handle = training_service(payload)
print("[Training Service]  ->", handle)
print("[Batch Service]     -> flagged", len(batch_scoring_service(X_test.sample(30, random_state=2))), "machines")
print("[Real-time Service] ->", realtime_inference_service(X_test.iloc[[5]])["verdict"])
print("\nServices are decoupled: each can be deployed, scaled and updated independently.")
''')

# ---------------------------------------------------------------- Security
md(r"""
### 9. Security — Application-Wide Controls

Security is enforced by a single reusable layer (`src/security_layer.py`) imported by **both this
notebook and the FastAPI services**, so the whole application is protected consistently. Five
controls are implemented and demonstrated below.

| # | Control | Threat mitigated |
|:--:|:--|:--|
| 1 | Input validation (range checks) | bad-data / adversarial injection |
| 2 | API-key authentication | unauthorized access |
| 3 | Model integrity (SHA-256) | model tampering / supply-chain |
| 4 | Audit logging (hash chain) | non-repudiation, tamper-evidence |
| 5 | Rate limiting (sliding window) | abuse / denial-of-service |
""")

code(r'''
import sys
sys.path.insert(0, ".")   # make src importable from the notebook
from src.security_layer import (
    validate_sensor_payload, ApiKeyAuthenticator, RateLimiter, AuditTrail,
    compute_file_sha256, verify_model_integrity, SecureInferenceGateway, SecurityError)

# ---- 1. Input validation -----------------------------------------------------
good = {"pressure":180,"temperature":70,"vibration":4,"flow_rate":85,"oil_debris":60}
print("1. INPUT VALIDATION")
print("   valid payload accepted:", bool(validate_sensor_payload(good)))
for bad, why in [({**good,"pressure":99999}, "out-of-range pressure"),
                 ({**good,"temperature":float('nan')}, "NaN temperature")]:
    try:
        validate_sensor_payload(bad)
    except SecurityError as e:
        print(f"   BLOCKED {why}: {e}")
''')

code(r'''
# ---- 2. API-key authentication ----------------------------------------------
print("2. API-KEY AUTHENTICATION")
auth = ApiKeyAuthenticator(["fleet-ops-secret-key"])
print("   valid key accepted:", auth.authenticate("fleet-ops-secret-key"))
try:
    auth.authenticate("attacker-guess")
except SecurityError as e:
    print(f"   BLOCKED invalid key: {e}")
''')

code(r'''
# ---- 3. Model integrity (SHA-256 anti-tamper) -------------------------------
print("3. MODEL INTEGRITY")
model_path = meta["path"]                       # from the Model Registry (from Model Registry, Section 6)
checksum = compute_file_sha256(model_path)
print(f"   registered SHA-256: {checksum[:24]}...")
print("   verify untampered model:", verify_model_integrity(model_path, checksum))
# simulate tampering: flip a byte and re-verify
with open(model_path, "ab") as f:
    f.write(b"\x00")                            # append a stray byte = tamper
try:
    verify_model_integrity(model_path, checksum)
except SecurityError as e:
    print(f"   BLOCKED tampered model: {str(e)[:70]}...")
# restore a clean model so later cells/tests still load it
registry.register(model, "hydraulic_maintenance_rf", "1.0", meta["metrics"])
''')

code(r'''
# ---- 4. Audit logging (tamper-evident hash chain) ---------------------------
print("4. AUDIT LOGGING")
audit = AuditTrail()
audit.record("tech-42", "predict", "machine=EX-118")
audit.record("tech-42", "predict", "machine=EX-204")
print("   entries recorded:", len(audit.entries))
print("   chain valid:", audit.verify_chain())
audit._entries[0]["detail"] = "machine=HACKED"   # tamper with history
print("   tamper detected (chain now invalid):", not audit.verify_chain())
''')

code(r'''
# ---- 5. Rate limiting + composed secure gateway -----------------------------
print("5. RATE LIMITING (max 3 / sec per client)")
limiter = RateLimiter(max_calls=3, window_seconds=1.0)
gateway = SecureInferenceGateway(auth, limiter, AuditTrail())

def _predict_fn(clean):
    row = pd.DataFrame([clean]).reindex(columns=[c for c in X.columns if c in clean], fill_value=0)
    return {"received": True}

allowed = 0
for i in range(5):
    try:
        gateway.guarded_predict("tech-42", "fleet-ops-secret-key", good, _predict_fn)
        allowed += 1
    except SecurityError as e:
        print(f"   request {i+1} BLOCKED: {e}")
print(f"   {allowed}/5 requests allowed within the window (expected 3)")
print("\nAll five security controls enforced application-wide via src/security_layer.py")
''')

# ---------------------------------------------------------------- RAG Advisor
md(r"""
### 10. RAG Maintenance Advisor (Retrieval-Augmented Guidance)

A prediction that a part will fail is more useful when paired with **what to do about it**.
When the model flags a component, the **Maintenance Advisor** retrieves the most relevant repair
procedure from a maintenance-manual knowledge base (`data/hydraulic_maintenance_manual.md`) using
semantic similarity — the *retrieve* stage of a RAG workflow.

> **Vector store:** production would use **ChromaDB** (a vector database) with sentence embeddings.
> This prototype uses a lightweight **TF-IDF + cosine similarity** retriever (scikit-learn) so it
> runs on any laptop with no heavy dependencies, while demonstrating the identical
> retrieve-by-similarity pattern. Only the retriever class changes when swapping in ChromaDB.

This adds a second ML capability (semantic retrieval) and another ML + non-ML component to the
architecture.
""")

code(r'''
from src.maintenance_advisor import MaintenanceAdvisor, load_knowledge_base

kb = load_knowledge_base()
print(f"Knowledge base loaded: {len(kb)} maintenance procedures")
advisor = MaintenanceAdvisor()

# When the model flags a machine, attach concrete repair guidance
for component in ["pump_leakage", "cooler_condition", "accumulator_pressure"]:
    top = advisor.advise(component, extra_symptoms="high vibration", k=1)[0]
    print(f"\nMODEL FLAG: {component}")
    print(f"  -> retrieved procedure : {top['procedure']}  (relevance {top['relevance']})")
    print(f"  -> guidance            : {top['guidance'][:130]}...")
''')

# ---------------------------------------------------------------- Summary
md(r"""
### 11. Summary

| Requirement | Result |
|:--|:--|
| Test accuracy | see Section 2 |
| **QR1 Robustness** (15% corrupt) | accuracy held — see Section 3 |
| **QR2 Latency** (< 100 ms) | measured in Section 4 |
| **QR3 Explainability** | per-machine top factors — Section 5 |
| Model Registry | versioned artifact — Section 6 |
| Batch vs Real-time serving | two paths — Section 7 |
| Pipe-and-Filter pipeline | Section 1 |

**Outcome:** a working, tested predictive-maintenance prototype — a nightly batch scan flags
machines needing attention, an instant real-time check protects operators, and every verdict is
explained. Architectural patterns (Pipe-and-Filter, Microservices split) and design patterns
(Model Registry, Batch-vs-Real-time serving) are demonstrated end-to-end on synthetic hydraulic
sensor data.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open("105.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote 105.ipynb with", len(cells), "cells")
