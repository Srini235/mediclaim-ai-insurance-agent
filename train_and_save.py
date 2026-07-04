"""
train_and_save.py — train the hydraulic predictive-maintenance models from the fleet
telemetry dataset and save them for the API server.

Author: Aman Kushwah (2024AC05064) — Group 105

Trains, matching the master notebook:
  * multi-output condition model  -> cooler_condition, valve_condition, pump_leakage, accumulator_pressure
  * lightweight stability model   -> stability_flag (real-time)

Run:  python3 train_and_save.py
Output: model_registry/condition_model.joblib, stability_model.joblib, schema.json
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score

DATA = "data/hydraulic_fleet_telemetry.csv"
REGISTRY = "model_registry"

NUMERIC_FEATURES = ["operating_hours", "pressure_mean_bar", "pressure_std_bar", "flow_mean_lpm",
                    "oil_temp_mean_c", "vibration_rms_mms", "motor_power_kw",
                    "pump_speed_mean_rpm", "cooling_efficiency_pct"]
CATEGORICAL_FEATURES = ["machine_type"]
TARGETS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]
RT_TARGET = "stability_flag"
NOISY_COLS = ["pressure_mean_bar", "flow_mean_lpm", "oil_temp_mean_c", "vibration_rms_mms"]


def main():
    os.makedirs(REGISTRY, exist_ok=True)
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"{DATA} missing — run merge_notebook.py to export the dataset.")
    df = pd.read_csv(DATA)

    # Data cleaning: median-impute any missing sensor readings
    imputer = SimpleImputer(strategy="median")
    df[NOISY_COLS] = imputer.fit_transform(df[NOISY_COLS])

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_multi = df[TARGETS]
    y_rt = df[RT_TARGET]
    X_tr, X_te, ym_tr, ym_te, yr_tr, yr_te = train_test_split(
        X, y_multi, y_rt, test_size=0.2, random_state=42, stratify=df[RT_TARGET])

    # 1. Multi-output condition model (batch)
    condition_model = Pipeline([
        ("prep", preprocessor),
        ("clf", MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)))])
    condition_model.fit(X_tr, ym_tr)
    ym_pred = condition_model.predict(X_te)
    cond_metrics = {t: {"accuracy": round(float(accuracy_score(ym_te[t], ym_pred[:, i])), 4),
                        "macro_f1": round(float(f1_score(ym_te[t], ym_pred[:, i], average="macro")), 4)}
                    for i, t in enumerate(TARGETS)}

    # 2. Lightweight stability model (real-time)
    stability_model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1))])
    stability_model.fit(X_tr, yr_tr)
    rt_acc = accuracy_score(yr_te, stability_model.predict(X_te))

    joblib.dump(condition_model, os.path.join(REGISTRY, "condition_model.joblib"))
    joblib.dump(stability_model, os.path.join(REGISTRY, "stability_model.joblib"))
    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "machine_types": sorted(df["machine_type"].unique().tolist()),
        "targets": TARGETS,
        "rt_target": RT_TARGET,
        # healthiest class per target (used to decide "flagged")
        "healthy_class": {"cooler_condition": 100, "valve_condition": 100,
                          "pump_leakage": 0, "accumulator_pressure": 130},
        "condition_metrics": cond_metrics,
        "stability_accuracy": round(float(rt_acc), 4),
    }
    with open(os.path.join(REGISTRY, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print("Saved condition_model.joblib, stability_model.joblib, schema.json")
    for t, m in cond_metrics.items():
        print(f"  {t:22s} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
    print(f"  stability_flag         acc={rt_acc:.3f}")


if __name__ == "__main__":
    main()
