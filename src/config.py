"""Configuration, seeds, disease catalog, and generation parameters.

All outputs of this pipeline are SYNTHETIC and intended only for research,
education, and methodological development. Not for clinical decision-making.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
GENERATOR_VERSION = "1.2.0"
GENERATOR_NAME = "SyntheticChronicDiseaseGenerator"

# ---------------------------------------------------------------------------
# Cohort size (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_N_PATIENTS = 3000

# Observation window (days from index)
MIN_OBS_DAYS = 180  # 6 months
MAX_OBS_DAYS = 3650  # 10 years

# Encounters
MIN_FOLLOWUPS = 0
MAX_FOLLOWUPS = 12

# Partition fractions (by patient ID)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Optional temporal / external-style holdout (subset of test)
EXTERNAL_HOLDOUT_FRAC_OF_TEST = 0.40

# ---------------------------------------------------------------------------
# Disease catalog
# ---------------------------------------------------------------------------
DISEASES = [
    "t2dm",
    "ckd",
    "cad",
    "hf",
    "htn",
    "copd",
    "cld_masld",
    "ra",
    "hypothyroid",
    "alzheimers",
]

DISEASE_LABELS = {
    "t2dm": "Type 2 diabetes mellitus",
    "ckd": "Chronic kidney disease",
    "cad": "Coronary artery disease",
    "hf": "Heart failure",
    "htn": "Hypertension",
    "copd": "Chronic obstructive pulmonary disease",
    "cld_masld": "Chronic liver disease / MASLD",
    "ra": "Rheumatoid arthritis",
    "hypothyroid": "Hypothyroidism",
    "alzheimers": "Alzheimer’s disease / related cognitive decline",
}

# Approximate adult prevalence targets (synthetic soft targets; not census rates).
# Values are deliberately somewhat higher than crude population rates so that
# after at-risk-only paths, delayed onset, and severity thresholds, the realized
# cohort still contains usable positive counts for ML development.
BASE_LIFETIME_RISK = {
    "t2dm": 0.22,
    "ckd": 0.16,
    "cad": 0.14,
    "hf": 0.09,
    "htn": 0.42,
    "copd": 0.12,
    "cld_masld": 0.24,
    "ra": 0.035,  # modest enrichment vs population ~1% for learnability
    "hypothyroid": 0.10,
    "alzheimers": 0.08,  # age-skewed in generator
}

# Stage labels
STAGE_LABELS = {
    "t2dm": ["none", "prediabetes", "diabetes_controlled", "diabetes_uncontrolled", "diabetes_with_complications"],
    "ckd": ["none", "stage_1_2", "stage_3a", "stage_3b", "stage_4_5"],
    "cad": ["none", "subclinical", "stable_cad", "prior_acs", "advanced_cad"],
    "hf": ["none", "stage_a_b", "hf_nyha_1_2", "hf_nyha_3", "hf_nyha_4"],
    "htn": ["none", "elevated", "stage_1", "stage_2", "resistant"],
    "copd": ["none", "gold_1", "gold_2", "gold_3", "gold_4"],
    "cld_masld": ["none", "steatosis", "steatohepatitis", "fibrosis", "cirrhosis"],
    "ra": ["none", "early", "moderate", "severe", "severe_erosive"],
    "hypothyroid": ["none", "subclinical", "overt_controlled", "overt_uncontrolled", "myxedema_risk"],
    "alzheimers": ["none", "subjective_decline", "mci", "mild_dementia", "moderate_severe_dementia"],
}

# Common multimorbidity edges (increases joint susceptibility)
COMORBIDITY_BOOST = [
    ("t2dm", "htn", 0.35),
    ("t2dm", "ckd", 0.40),
    ("t2dm", "cad", 0.30),
    ("t2dm", "cld_masld", 0.35),
    ("htn", "ckd", 0.35),
    ("htn", "cad", 0.30),
    ("htn", "hf", 0.40),
    ("cad", "hf", 0.45),
    ("copd", "cad", 0.20),
    ("copd", "hf", 0.25),
    ("hypothyroid", "t2dm", 0.10),
    ("ra", "cad", 0.15),
    ("cld_masld", "t2dm", 0.30),
    ("ckd", "hf", 0.25),
]

# Medication classes modeled
MEDICATION_CLASSES = [
    "metformin",
    "sglt2i",
    "glp1ra",
    "insulin",
    "acei_arb",
    "beta_blocker",
    "statin",
    "antiplatelet",
    "loop_diuretic",
    "thiazide",
    "calcium_channel_blocker",
    "inhaled_bronchodilator",
    "inhaled_corticosteroid",
    "systemic_corticosteroid",
    "dmard_methotrexate",
    "biologic_dmard",
    "levothyroxine",
    "ppi",
    "nsaid",
]

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
NOISY_DIR = DATA_DIR / "noisy"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Demographics (synthetic research cohort; not a real census)
SEX_PROBS = {"female": 0.52, "male": 0.48}
ETHNICITY = {
    "White": 0.55,
    "Black_or_African_American": 0.15,
    "Hispanic_or_Latino": 0.15,
    "Asian": 0.08,
    "Other_or_Multiple": 0.07,
}
SES = {"low": 0.28, "medium": 0.47, "high": 0.25}
ACCESS = {"limited": 0.22, "standard": 0.58, "enhanced": 0.20}
REGION = {"urban": 0.45, "suburban": 0.35, "rural": 0.20}

SMOKING = {"never": 0.55, "former": 0.28, "current": 0.17}
ALCOHOL = {"none": 0.30, "moderate": 0.55, "heavy": 0.15}
ACTIVITY = {"sedentary": 0.35, "moderate": 0.45, "active": 0.20}
DIET_RISK = {"low": 0.25, "moderate": 0.50, "high": 0.25}

# Sensitive / audit variables (retained for fairness auditing)
SENSITIVE_AUDIT_VARS = [
    "sex_at_birth",
    "gender",
    "ethnicity",
    "ses_group",
    "access_to_care",
    "region",
    "age_at_index",
]
