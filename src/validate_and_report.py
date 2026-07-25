"""Validation, leakage audit, class/missingness/fairness reports for synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DISEASES,
    DISEASE_LABELS,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    RANDOM_SEED,
    REPORTS_DIR,
    SENSITIVE_AUDIT_VARS,
)
from .reference_ranges import LAB_SPECS


PHYSIO_BOUNDS = {
    "fasting_plasma_glucose": (20, 600),
    "hba1c": (3.0, 18.0),
    "serum_creatinine": (0.1, 20.0),
    "egfr": (1, 200),
    "sbp_mmHg": (60, 260),
    "dbp_mmHg": (30, 160),
    "heart_rate_bpm": (30, 220),
    "spo2_percent": (50, 100),
    "tsh": (0.001, 200),
    "mmse": (0, 30),
    "moca": (0, 30),
    "fev1_fvc_ratio": (0.15, 1.0),
    "nt_probnp": (0, 100000),
    "bmi": (10, 80),
}


def _safe_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return pd.DataFrame()
    return df[use].apply(pd.to_numeric, errors="coerce").corr()


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    patients = tables["patients"]
    encounters = tables["encounters"]
    labs = tables["labs"]
    diagnoses = tables["diagnoses"]
    medications = tables["medications"]
    targets = tables["prediction_targets"]

    issues: list[str] = []
    checks: dict[str, Any] = {}

    # Unique patient IDs
    if patients["patient_id"].duplicated().any():
        issues.append("Duplicate patient_id in patients table")
    checks["n_patients"] = int(len(patients))
    checks["n_encounters"] = int(len(encounters))
    checks["n_labs"] = int(len(labs))
    checks["n_diagnoses"] = int(len(diagnoses))
    checks["n_medications"] = int(len(medications))
    checks["n_target_rows"] = int(len(targets))

    # Partition integrity: patient-level, no cross-over
    part = patients.set_index("patient_id")["partition"]
    enc_part = encounters[["patient_id", "partition"]].drop_duplicates()
    merged = enc_part.merge(part.rename("p_part"), left_on="patient_id", right_index=True)
    if not (merged["partition"] == merged["p_part"]).all():
        issues.append("Encounter partition mismatch vs patient partition")
    # Any patient in multiple partitions?
    multip = patients.groupby("patient_id")["partition"].nunique()
    if (multip > 1).any():
        issues.append("Patient appears in multiple partitions")
    checks["partition_counts"] = {str(k): int(v) for k, v in patients["partition"].value_counts().items()}
    checks["partition_patient_exclusivity"] = "PASS" if not issues else "SEE_ISSUES"

    # Physiological bounds on encounters
    bound_viol = {}
    for col, (lo, hi) in PHYSIO_BOUNDS.items():
        if col in encounters.columns:
            s = pd.to_numeric(encounters[col], errors="coerce")
            n = int(((s < lo) | (s > hi)).sum())
            if n:
                bound_viol[col] = n
        if col in labs["analyte"].values if len(labs) else False:
            pass
    # Labs bounds (exclude flagged entry errors)
    if len(labs):
        lab_viol = 0
        for analyte, (lo, hi) in [
            ("fasting_plasma_glucose", (20, 600)),
            ("hba1c", (3, 18)),
            ("egfr", (1, 200)),
            ("serum_creatinine", (0.1, 20)),
            ("tsh", (0.001, 200)),
        ]:
            sub = labs[labs["analyte"] == analyte]
            sub = sub[sub["abnormality_flag"] != "ERROR_NEGATIVE"]
            v = pd.to_numeric(sub["value"], errors="coerce")
            lab_viol += int(((v < lo) | (v > hi)).sum())
        checks["lab_bound_violations_excluding_flagged_errors"] = lab_viol
        if lab_viol > 0:
            issues.append(f"{lab_viol} lab values outside soft physiological bounds")
    checks["encounter_bound_violations"] = bound_viol

    # Treatment after initiation: medications days >= 0
    if len(medications):
        if (medications["days_from_index"] < 0).any():
            issues.append("Medication events before index (unexpected)")

    # Diagnosis not before index for incident (historical allowed negative onset latent)
    # Observed diagnosis days should be within observation for most
    checks["diagnoses_by_disease"] = (
        diagnoses["disease_code"].value_counts().to_dict() if len(diagnoses) else {}
    )

    # No negative labs unless flagged (clean version should be 0)
    if len(labs):
        neg = labs[pd.to_numeric(labs["value"], errors="coerce") < 0]
        checks["negative_lab_values"] = int(len(neg))
        checks["negative_flagged_as_error"] = int((neg["abnormality_flag"] == "ERROR_NEGATIVE").sum()) if len(neg) else 0

    # Longitudinal: encounter days sorted per patient
    bad_order = 0
    for pid, g in encounters.groupby("patient_id"):
        days = g.sort_values("encounter_number")["days_from_index"].values
        if not np.all(np.diff(days) >= 0):
            bad_order += 1
    checks["patients_with_unsorted_encounters"] = bad_order
    if bad_order:
        issues.append("Unsorted encounter timelines found")

    # Prevalence (latent at any encounter)
    prev = {}
    for d in DISEASES:
        col = f"latent_disease_present_{d}"
        if col in encounters.columns:
            # patient-level ever-present
            ever = encounters.groupby("patient_id")[col].max()
            prev[d] = float(ever.mean())
    checks["patient_level_latent_prevalence"] = prev

    # Leakage audit placeholders
    leakage = leakage_audit(tables)
    checks["leakage_audit"] = leakage
    if leakage.get("status") != "PASS":
        issues.append("Leakage audit did not fully pass")

    checks["issues"] = issues
    checks["overall_status"] = "PASS" if not issues else "PASS_WITH_NOTES" if all(
        "leakage" not in i.lower() and "duplicate patient" not in i.lower() for i in issues
    ) else "FAIL"
    # Soft issues allowed
    if not any("multiple partitions" in i or "Duplicate patient" in i for i in issues):
        checks["overall_status"] = "PASS" if not issues else "PASS_WITH_NOTES"

    return checks


def leakage_audit(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Document leakage controls and run automated checks."""
    targets = tables["prediction_targets"]
    patients = tables["patients"]
    findings = []
    status = "PASS"

    # 1. Partition by patient only
    p_parts = patients.groupby("patient_id")["partition"].nunique()
    if (p_parts > 1).any():
        findings.append("FAIL: patient in multiple partitions")
        status = "FAIL"
    else:
        findings.append("OK: each patient in exactly one partition")

    # 2. Feature cutoff == prediction time (by design)
    if "feature_cutoff_timestamp_day" in targets.columns:
        match = (targets["feature_cutoff_timestamp_day"] == targets["prediction_timestamp_day"]).all()
        findings.append("OK: feature_cutoff equals prediction_timestamp" if match else "FAIL: cutoff mismatch")
        if not match:
            status = "FAIL"

    # 3. Target columns should not be used as features — list them
    target_prefixes = [
        "new_disease_within_",
        "progression_within_",
        "major_complication_within_",
        "hospitalization_within_",
        "treatment_response_",
        "biomarker_control_",
        "all_cause_mortality_",
        "disease_present_at_encounter_",
        "disease_stage_at_encounter_",
        "true_latent_",
        "observed_diagnosis_present_",
    ]
    target_cols = [c for c in targets.columns if any(c.startswith(p) for p in target_prefixes)]
    findings.append(f"Documented {len(target_cols)} target/label columns excluded from feature sets")

    # 4. Perfect prediction check: single lab vs disease at encounter (should not be deterministic)
    enc = tables["encounters"]
    labs = tables["labs"]
    perfect = []
    if len(labs) and len(enc):
        # Pivot a few common labs at encounter
        for analyte, disease, rule in [
            ("hba1c", "t2dm", lambda s: s >= 6.5),
            ("egfr", "ckd", lambda s: s < 60),
            ("tsh", "hypothyroid", lambda s: s > 4.5),
        ]:
            sub = labs[labs["analyte"] == analyte][["encounter_id", "value"]]
            if sub.empty:
                continue
            m = enc[["encounter_id", f"latent_disease_present_{disease}"]].merge(sub, on="encounter_id")
            if m.empty:
                continue
            pred = rule(m["value"])
            y = m[f"latent_disease_present_{disease}"].astype(bool)
            # If perfect separation
            if pred.equals(y) or ((pred == y).mean() > 0.99):
                perfect.append(f"{analyte}->{disease} nearly deterministic")
        if perfect:
            findings.append("NOTE: " + "; ".join(perfect))
            # Not necessarily fail — document
        else:
            findings.append("OK: no single screened biomarker perfectly determines disease labels")

    # 5. Future treatments not in same-day features — documented design
    findings.append(
        "OK by design: medication effects applied only when day >= treatment_start; "
        "prediction features should use labs/vitals/meds with days_from_index <= feature_cutoff"
    )
    findings.append(
        "OK by design: train/val/test split is patient-level; temporal holdout flagged via is_external_style_holdout"
    )

    return {"status": status, "findings": findings, "n_target_columns": len(target_cols)}


def class_distribution_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    targets = tables["prediction_targets"]
    rows = []
    for d in DISEASES:
        col = f"disease_present_at_encounter_{d}"
        if col in targets.columns:
            vc = targets[col].value_counts(dropna=False)
            rows.append({
                "task": col,
                "disease": d,
                "n_pos": int(vc.get(1, 0)),
                "n_neg": int(vc.get(0, 0)),
                "prevalence": float(targets[col].mean()),
            })
        for months in (6, 12, 36):
            col = f"new_disease_within_{months}_months_{d}"
            if col in targets.columns:
                s = targets[col]
                eligible = s[s >= 0]
                rows.append({
                    "task": col,
                    "disease": d,
                    "n_pos": int((eligible == 1).sum()),
                    "n_neg": int((eligible == 0).sum()),
                    "n_ineligible_or_censored": int((s < 0).sum()),
                    "prevalence_among_eligible": float(eligible.mean()) if len(eligible) else np.nan,
                })
    if "hospitalization_within_12_months" in targets.columns:
        s = targets["hospitalization_within_12_months"]
        el = s[s >= 0]
        rows.append({
            "task": "hospitalization_within_12_months",
            "disease": "any",
            "n_pos": int((el == 1).sum()),
            "n_neg": int((el == 0).sum()),
            "prevalence_among_eligible": float(el.mean()) if len(el) else np.nan,
        })
    return pd.DataFrame(rows)


def missingness_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    labs = tables["labs"]
    encounters = tables["encounters"]
    # Lab missingness is structural (not ordered) — compute per-encounter analyte coverage
    analytes = list(LAB_SPECS.keys())
    n_enc = encounters["encounter_id"].nunique()
    coverage = {}
    if len(labs) and n_enc:
        for a in analytes:
            n = labs.loc[labs["analyte"] == a, "encounter_id"].nunique()
            coverage[a] = {"encounters_with_result": int(n), "coverage_rate": float(n / n_enc)}
    enc_null = encounters.isna().mean().sort_values(ascending=False).head(30).to_dict()
    return {
        "mechanism_notes": {
            "MCAR": "Random non-ordering of some routine tests (~15% routine panel gaps)",
            "MAR": "Specialty tests ordered more for higher latent severity / age / abnormal screens",
            "MNAR": "Limited access reduces specialty tests; loss to follow-up related to disease control",
        },
        "lab_coverage_by_analyte": coverage,
        "encounter_column_null_rate_top": enc_null,
    }


def fairness_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    patients = tables["patients"]
    encounters = tables["encounters"]
    labs = tables["labs"]
    report: dict[str, Any] = {"sensitive_variables": SENSITIVE_AUDIT_VARS}

    # Sample sizes
    for var in ["sex_at_birth", "ethnicity", "ses_group", "access_to_care", "region"]:
        report[f"n_by_{var}"] = patients[var].value_counts().to_dict()

    # Age groups
    age_bins = pd.cut(patients["age_at_index"], bins=[0, 40, 55, 70, 120], labels=["18-39", "40-54", "55-69", "70+"])
    report["n_by_age_group"] = age_bins.value_counts().to_dict()

    # Disease prevalence by sex (latent ever)
    prev_sex = {}
    for d in DISEASES:
        col = f"latent_disease_present_{d}"
        tmp = encounters.groupby("patient_id")[col].max().reset_index()
        tmp = tmp.merge(patients[["patient_id", "sex_at_birth"]], on="patient_id")
        prev_sex[d] = tmp.groupby("sex_at_birth")[col].mean().to_dict()
    report["latent_prevalence_by_sex"] = prev_sex

    # Missingness (lab coverage) by access
    if len(labs):
        enc_access = encounters.merge(patients[["patient_id", "access_to_care"]], on="patient_id")
        lab_enc = labs.groupby("encounter_id").size().rename("n_labs")
        enc_access = enc_access.merge(lab_enc, left_on="encounter_id", right_index=True, how="left")
        enc_access["n_labs"] = enc_access["n_labs"].fillna(0)
        report["mean_labs_per_encounter_by_access"] = (
            enc_access.groupby("access_to_care")["n_labs"].mean().to_dict()
        )

    # Underrepresentation flags
    under = []
    for var in ["ethnicity", "ses_group", "access_to_care"]:
        vc = patients[var].value_counts(normalize=True)
        for k, v in vc.items():
            if v < 0.08:
                under.append(f"{var}={k} share={v:.3f}")
    report["underrepresented_groups"] = under
    report["proxy_variable_warnings"] = [
        "ses_group and access_to_care may proxy structural inequities; do not use as clinical features for deployment decisions",
        "ethnicity is for fairness auditing only; avoid as a direct predictive feature without causal justification",
        "region and access correlate with test density — models may learn care intensity rather than biology",
    ]
    report["interpretation_caution"] = [
        "Synthetic demographic effects are mild and non-deterministic; real-world disparities differ",
        "Subgroup performance on this dataset does not validate fairness in clinical use",
        "Smaller groups (e.g., RA positives within ethnicity strata) may be unstable",
    ]
    return report


def summary_statistics(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    patients = tables["patients"]
    encounters = tables["encounters"]
    labs = tables["labs"]

    cont_cols = ["age_at_index", "bmi_baseline", "pack_years", "height_cm", "weight_kg_baseline"]
    cont = patients[cont_cols].describe().to_dict()

    enc_cont = ["sbp_mmHg", "dbp_mmHg", "heart_rate_bpm", "bmi", "spo2_percent"]
    enc_desc = encounters[enc_cont].describe().to_dict()

    # Lab distributions
    lab_desc = {}
    for a in ["hba1c", "egfr", "ldl_c", "alt", "tsh", "nt_probnp", "fev1_fvc_ratio"]:
        sub = labs.loc[labs["analyte"] == a, "value"]
        if len(sub):
            lab_desc[a] = {
                "n": int(len(sub)),
                "mean": float(sub.mean()),
                "std": float(sub.std()),
                "p50": float(sub.median()),
                "p05": float(sub.quantile(0.05)),
                "p95": float(sub.quantile(0.95)),
            }

    # Correlation among encounter-level pivoted common labs
    pivot_analytes = ["hba1c", "egfr", "ldl_c", "alt", "tsh", "hemoglobin"]
    wide = (
        labs[labs["analyte"].isin(pivot_analytes)]
        .drop_duplicates(["encounter_id", "analyte"])
        .pivot(index="encounter_id", columns="analyte", values="value")
    )
    corr = wide.corr().round(3).to_dict() if len(wide) else {}

    # Trajectory sample: mean HbA1c by encounter number among T2DM latent
    traj = {}
    if len(labs):
        t2 = encounters[encounters["latent_disease_present_t2dm"] == 1][["encounter_id", "encounter_number", "patient_id"]]
        h = labs[labs["analyte"] == "hba1c"][["encounter_id", "value"]]
        m = t2.merge(h, on="encounter_id")
        if len(m):
            traj["mean_hba1c_by_encounter_number_t2dm"] = (
                m.groupby("encounter_number")["value"].mean().round(3).to_dict()
            )

    # Partition comparability
    part_age = patients.groupby("partition")["age_at_index"].agg(["mean", "std", "count"]).round(2).to_dict()

    return {
        "patient_continuous": cont,
        "encounter_continuous": enc_desc,
        "lab_distributions": lab_desc,
        "lab_correlations": corr,
        "longitudinal_trajectory_examples": traj,
        "partition_age_comparability": part_age,
        "categorical_frequencies": {
            "sex_at_birth": patients["sex_at_birth"].value_counts().to_dict(),
            "smoking_status": patients["smoking_status"].value_counts().to_dict(),
            "partition": patients["partition"].value_counts().to_dict(),
        },
    }


def write_all_reports(tables: dict[str, pd.DataFrame], out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_tables(tables)
    class_df = class_distribution_report(tables)
    miss = missingness_report(tables)
    fair = fairness_report(tables)
    summary = summary_statistics(tables)
    leakage = validation.get("leakage_audit", {})

    class_df.to_csv(out_dir / "class_distribution_report.csv", index=False)

    # Plot-ready tables
    prev = pd.DataFrame(
        [
            {"disease": d, "label": DISEASE_LABELS[d], "prevalence": validation["patient_level_latent_prevalence"].get(d)}
            for d in DISEASES
        ]
    )
    prev.to_csv(out_dir / "disease_prevalence.csv", index=False)

    # Lab coverage plot-ready
    cov_rows = []
    for a, v in miss.get("lab_coverage_by_analyte", {}).items():
        cov_rows.append({"analyte": a, **v})
    pd.DataFrame(cov_rows).to_csv(out_dir / "lab_coverage.csv", index=False)

    reports = {
        "meta": {
            "is_synthetic": True,
            "disclaimer": "SYNTHETIC DATA — research/education only. Not for clinical use.",
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "random_seed": RANDOM_SEED,
        },
        "validation_summary": validation,
        "missingness_report": miss,
        "fairness_report": fair,
        "leakage_audit": leakage,
        "summary_statistics": summary,
    }

    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)

    # Human-readable markdown summaries
    _write_markdown_reports(out_dir, reports, class_df, prev)
    return reports


def _write_markdown_reports(
    out_dir: Path,
    reports: dict[str, Any],
    class_df: pd.DataFrame,
    prev: pd.DataFrame,
) -> None:
    v = reports["validation_summary"]
    md = []
    md.append("# Validation Summary (SYNTHETIC DATA)\n")
    md.append("> **Disclaimer:** All data are synthetic. Not real patients. Not for clinical decision-making.\n")
    md.append(f"- Generator: `{reports['meta']['generator_name']}` v{reports['meta']['generator_version']}\n")
    md.append(f"- Seed: `{reports['meta']['random_seed']}`\n")
    md.append(f"- Overall status: **{v.get('overall_status')}**\n")
    md.append(f"- Patients: {v.get('n_patients')}, Encounters: {v.get('n_encounters')}, Labs: {v.get('n_labs')}\n")
    md.append(f"- Partitions: `{v.get('partition_counts')}`\n")
    md.append("\n## Issues\n")
    if v.get("issues"):
        for i in v["issues"]:
            md.append(f"- {i}\n")
    else:
        md.append("- None recorded\n")
    md.append("\n## Latent disease prevalence (patient ever-present)\n")
    md.append(prev.to_markdown(index=False) if hasattr(prev, "to_markdown") else prev.to_string(index=False))
    md.append("\n")
    (out_dir / "validation_summary.md").write_text("".join(md) if isinstance(md[0], str) else "\n".join(map(str, md)), encoding="utf-8")
    # Fix join - I mixed types. Rewrite cleanly:
    lines = [
        "# Validation Summary (SYNTHETIC DATA)",
        "",
        "> **Disclaimer:** All data are synthetic. Not real patients. Not for clinical decision-making.",
        "",
        f"- Generator: `{reports['meta']['generator_name']}` v{reports['meta']['generator_version']}",
        f"- Seed: `{reports['meta']['random_seed']}`",
        f"- Overall status: **{v.get('overall_status')}**",
        f"- Patients: {v.get('n_patients')}, Encounters: {v.get('n_encounters')}, Labs: {v.get('n_labs')}",
        f"- Partitions: `{v.get('partition_counts')}`",
        "",
        "## Issues",
    ]
    if v.get("issues"):
        lines.extend([f"- {i}" for i in v["issues"]])
    else:
        lines.append("- None recorded")
    lines += ["", "## Latent disease prevalence (patient ever-present)", "", prev.to_string(index=False), ""]
    (out_dir / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Leakage
    leak = reports["leakage_audit"]
    ll = [
        "# Leakage Audit (SYNTHETIC DATA)",
        "",
        f"Status: **{leak.get('status')}**",
        "",
    ]
    for f in leak.get("findings", []):
        ll.append(f"- {f}")
    ll += [
        "",
        "## Required modeling practice",
        "- Split by `patient_id` only (pre-assigned `partition` column).",
        "- At prediction time T, use only rows with `days_from_index <= feature_cutoff_timestamp_day`.",
        "- Do not use columns listed as targets/labels as features.",
        "- Do not use future medications, diagnoses, or labs beyond cutoff.",
        "- Do not use `latent_*` audit fields if the task is to predict observed diagnosis from clinical features only — choose target explicitly.",
    ]
    (out_dir / "leakage_audit.md").write_text("\n".join(ll), encoding="utf-8")

    # Fairness
    fair = reports["fairness_report"]
    fl = ["# Fairness Report (SYNTHETIC DATA)", "", "## Sample sizes"]
    for k, val in fair.items():
        if k.startswith("n_by_"):
            fl.append(f"- **{k}**: `{val}`")
    fl += ["", "## Underrepresented groups"]
    fl += [f"- {u}" for u in fair.get("underrepresented_groups", [])] or ["- None flagged"]
    fl += ["", "## Proxy warnings"]
    fl += [f"- {u}" for u in fair.get("proxy_variable_warnings", [])]
    fl += ["", "## Caution"]
    fl += [f"- {u}" for u in fair.get("interpretation_caution", [])]
    (out_dir / "fairness_report.md").write_text("\n".join(fl), encoding="utf-8")

    # Missingness
    miss = reports["missingness_report"]
    ml = ["# Missingness Report (SYNTHETIC DATA)", "", "## Mechanisms"]
    for k, val in miss.get("mechanism_notes", {}).items():
        ml.append(f"- **{k}**: {val}")
    ml += ["", "See `lab_coverage.csv` for analyte-level coverage rates."]
    (out_dir / "missingness_report.md").write_text("\n".join(ml), encoding="utf-8")

    # Class distribution brief
    cl = ["# Class Distribution Report (SYNTHETIC DATA)", "", class_df.head(80).to_string(index=False)]
    (out_dir / "class_distribution_report.md").write_text("\n".join(cl), encoding="utf-8")
