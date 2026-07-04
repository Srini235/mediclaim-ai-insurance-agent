# Objective 1 — Requirements Formulation (GR4ML)
### Predictive Maintenance of Mobile Hydraulic Systems · Group 105

---

## 1. Domain and Problem Statement

Construction and mining fleets rely on excavators whose hydraulic systems — the pumps, valves,
cooler and accumulator that drive every dig, lift and swing — are today serviced on a **fixed
calendar schedule**. The intensity of use varies enormously between machines: a unit working a
mine face is stressed far harder than one doing light landscaping. A one-size-fits-all schedule
therefore fails in both directions. Healthy components are replaced too early, wasting parts and
labour, while other components fail *before* their scheduled check, causing an excavator to break
down mid-job — which is expensive, disrupts the site, and can be unsafe for the operator.

**Problem statement.** Build an ML-based system that continuously assesses the health of hydraulic
components from onboard sensor data (pressure, temperature, vibration, flow rate, oil debris) and
recommends maintenance **only when a component actually needs it**. The system must say *which*
component is at risk, *how confident* it is, and *why*, so a technician can act with the right spare
part instead of following a blind schedule.

**Measurable goals.**
- Predict component maintenance need with **≥ 85% accuracy / macro-F1** on held-out data.
- Return the safety-critical real-time assessment in **under 100 ms**.
- Remain accurate (**≥ 95% of clean accuracy retained**) when up to **15% of sensor readings are
  noisy or missing**.
- Provide a **human-readable explanation** (top contributing sensors) for every flagged machine.

---

## 2. GR4ML — Business View (*Why?*)

The Business View captures who the system serves and the goals/softgoals that justify it.

**Stakeholders and their goals**

| Stakeholder | Hard goal | Softgoal (quality) |
|:--|:--|:--|
| Maintenance Manager | Reduce total maintenance cost; avoid unplanned downtime | Cost-efficiency, reliability |
| Field Technician | Know exactly which component is failing before travelling | Actionability, explainability |
| Machine Operator | Be warned immediately if the machine is unsafe *now* | Safety, low latency |
| Fleet Owner (business) | Maximise machine availability and asset life | Availability, trust |

**Business goal decomposition**
- *Top goal:* “Minimise hydraulic maintenance cost while avoiding unplanned failures.”
  - *Sub-goal 1:* Predict failures early → enables condition-based (not calendar-based) service.
  - *Sub-goal 2:* Prioritise which machines to service → efficient technician scheduling.
  - *Sub-goal 3:* Protect operators in real time → instant unsafe-condition alerts.
- *Softgoals* (constrain how the goals are met): **Safety**, **Reliability/Robustness**,
  **Explainability/Trust**, **Low latency**, **Security**.

---

## 3. GR4ML — Analytics Design View (*What?*)

This view maps each business decision to the analytic question and the modelling choice that
answers it.

| Decision to support | Analytic question | Answer type | Timeliness |
|:--|:--|:--|:--|
| Which machines to service soon? | “Is this component degrading?” | Classification (maintenance / healthy) | Batch (overnight) |
| Is a machine unsafe right now? | “Is the machine in an unstable state?” | Classification + confidence | Real-time (<100 ms) |
| What should the technician do? | “What repair procedure fits this fault?” | Semantic retrieval (RAG) | On demand |

**Model choice and justification — Random Forest.**
Several approaches were considered:
- *Threshold rules* — transparent but too rigid for multi-sensor interactions and easily fooled by
  noise.
- *Deep neural network* — powerful but data-hungry, slower to infer, and hard to explain to a
  technician.
- **Random Forest (selected)** — strong accuracy on tabular sensor data, fast single-sample
  inference (well under the 100 ms budget), naturally robust to noisy/missing values, and
  **self-explaining** through feature importances. Explainability was the deciding factor: a
  technician must see *why* a machine was flagged, not accept a black-box verdict.

**Retrieval component.** When a component is flagged, a Retrieval-Augmented Generation (RAG) step
fetches the matching repair procedure from a maintenance-manual knowledge base by semantic
similarity, turning a prediction into actionable guidance.

---

## 4. GR4ML — Data Preparation View (*How?*)

Real hydraulic sensor streams are noisy — dust, vibration and electrical interference cause
occasional missing or spurious readings — so the data must be prepared before modelling. This is
implemented as a **Pipe-and-Filter** pipeline.

**Data sources.** Onboard hydraulic sensors per work-cycle: pressure (bar), temperature (°C),
vibration (mm/s RMS), flow rate (L/min), oil debris (ppm). Because real fleet logs are proprietary,
a **realistic synthetic generator** reproduces the physical relationships (heat, vibration and
debris rising with duty intensity, with a plausible failure relationship) so the full system can be
built and tested. The knowledge base for RAG is a maintenance manual document.

**Preparation stages (filters).**
1. **Ingest** — acquire raw per-cycle sensor readings.
2. **Clean** — detect and repair missing/glitched values (median imputation), as real dusty sensors
   require.
3. **Aggregate** — derive cross-sensor cycle summaries (e.g. thermal-load ratio, vibration-per-
   pressure) that expose degradation.
4. **Feature** — emit the final feature matrix for the model.

**Quality of data controls.** Range validation rejects physically-impossible readings before they
reach the model (also a security control), and the pipeline is explicitly tested against 15%
corruption to confirm downstream robustness.

---

## 5. Top-3 Quality Requirements and Justification

The three non-negotiable quality attributes were selected because each maps directly to a
stakeholder softgoal and to a way the system could fail in the field.

| # | Quality requirement | Target | Why it is critical | How it is verified |
|:--:|:--|:--|:--|:--|
| 1 | **Robustness to noisy data** | ≥ 95% of clean accuracy retained under 15% corrupted readings | Field sensors glitch constantly; a model that collapses on messy data is useless on real machines | Deliberately corrupt 15% of test data and re-measure accuracy (≈ 99% retained) |
| 2 | **Low latency (real-time)** | < 100 ms per safety inference | The operator-safety path is like a collision warning — a slow answer is no answer | Time single-sample inference over many warm calls (≈ 50 ms) |
| 3 | **Explainability** | Every flag names its top contributing sensors | Technicians must trust and act on the verdict; a black box will be ignored or mis-serviced | Report global feature importances and per-machine top factors |

*(A fourth attribute, **Security**, is also implemented — input validation, authentication, model
integrity, audit logging and rate limiting — and can be presented as an additional quality
requirement, since the system serves safety-critical fleet infrastructure.)*

Justification for the selection: **Safety** (QR2) and **Reliability** (QR1) are the softgoals with
the highest business and human cost if violated, and **Explainability** (QR3) is the softgoal that
determines whether the system is actually adopted by the people who must use it. Together they cover
the “does it work when it matters”, “does it work on real data”, and “will people trust it”
dimensions.

---

## 6. Traceability — GR4ML element → implementation

Every element above is demonstrated and executed in `105.ipynb`.

| GR4ML element | Where it is realised |
|:--|:--|
| Business View stakeholders/goals | Notebook GR4ML markdown; served by Real-time + Batch paths |
| Analytics Design — Random Forest | Section 2 (training, 5-fold CV, metrics) |
| Analytics Design — RAG retrieval | Section 10 RAG Maintenance Advisor |
| Data Preparation — Pipe-and-Filter | Section 1 (Ingest → Clean → Aggregate → Feature) |
| QR1 Robustness | Section 3 (15% corruption test) |
| QR2 Latency | Section 4 (inference timing) |
| QR3 Explainability | Section 5 (feature importances, per-machine factors) |
| Security (extra QR) | Section 9 + `src/security_layer.py` |
| System architecture (ML + non-ML) | Architecture diagram cell / `architecture_diagram.png` |
