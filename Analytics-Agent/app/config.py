"""
app/config.py — Central Configuration for the Analytics Agent

Unlike the other three services in this monorepo, this file deliberately
blends two roles the others keep separate (a pure-env `Settings` class in
`app/config.py`, YAML-driven agent config in `app/agents/<name>/config.py`):
Agent 3's business logic (`app/services/*.py`) is so uniformly config-driven
(KPI definitions, business rules, ML hyperparameters, feature columns) that
splitting the loader across two files would just mean every service imports
from two places for one coherent set of settings. `app/agents/analytics_agent/config.py`
still exists and still owns the LangGraph-specific accessor (`get_entry_point()`),
matching the other agents' structure — it just reads through this module
rather than re-parsing `agent.yaml` a second time.

Priority order for settings:
  1. app/agents/analytics_agent/agent.yaml — agent identity, role, memory, system prompt
  2. config/ml_config.yml                  — all ML model settings (swappable)
  3. config/business_rules.yml             — predefined insurance rules + new rules
  4. .env                                  — secrets (API keys), HOST/PORT
  5. ../.env (repo root)                   — AZURE_OPENAI_*/POSTGRES_*/LOG_LEVEL shared with
                                              Agent 1/2/Orchestrator, fallback only — this
                                              service's own .env always wins on conflicts
  6. Hardcoded defaults below              — last resort fallback
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Local .env first — load_dotenv() never overwrites a key already set, so
# loading local before root is what makes local win on any key both define.
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
# This file lives at app/config.py, one level deeper than the old top-level
# config.py — .parent.parent to get back to the Analytics-Agent repo root.
BASE_DIR   = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR   = BASE_DIR / "data"

# Root .env (AZURE_OPENAI_*, POSTGRES_*, LOG_LEVEL shared with Agent 1/2/
# Orchestrator) — loaded after local, so it only fills in keys local didn't
# already set. See this module's docstring for the full priority order.
load_dotenv(BASE_DIR.parent / ".env")
ML_DIR     = BASE_DIR / "ml"
RULES_DIR  = CONFIG_DIR / "rules"
AGENT_YAML_PATH = BASE_DIR / "app" / "agents" / "analytics_agent" / "agent.yaml"

# Primary dataset — only relevant to scripts/cli.py and train.py now; the
# FastAPI route (app/routes/analyze.py) always overrides this per-request
# with the uploaded file's own temp path. Default points at the shared
# repo-root test_data/ (see the root README) rather than a hardcoded
# personal-machine path — .env should still set this explicitly for real
# use, this is just a fallback that doesn't fail immediately on a fresh
# clone.
DATASET_PATH = Path(os.getenv(
    "DATASET_PATH",
    str(BASE_DIR.parent / "test_data" / "insurance_variance_data_native.csv"),
))

# ── Service host/port ──────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

# Same limit/behavior as Agent 1's own MAX_UPLOAD_SIZE_MB — /analyze reads
# the whole upload into memory before writing it to a temp file for DuckDB.
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

# ── Shared Postgres (conversation memory only — same instance/DB as Agent 1/2,
# own "agent3" schema) ────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "mva_pipeline")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# ── YAML Loaders ──────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

AGENT_CONFIG = _load_yaml(AGENT_YAML_PATH)
ML_CONFIG    = _load_yaml(CONFIG_DIR / "ml_config.yml")
RULES_CONFIG = _load_yaml(CONFIG_DIR / "business_rules.yml")
# Structural-prerequisite thresholds for Stage 1's AnalyticsCapabilityResolver
# (Agent 3 redesign) — dataset-shape checks, NOT readiness scoring. See the
# file's own header for why these are kept separate from ML_READINESS_THRESHOLD/
# LLM_READINESS_THRESHOLD below.
CAPABILITY_THRESHOLDS = _load_yaml(CONFIG_DIR / "capability_thresholds.yml")
# Stage 5's AnalyticsScheduler budget defaults (Agent 3 redesign) — see the
# file's own header for the full rationale.
SCHEDULING_BUDGET = _load_yaml(CONFIG_DIR / "scheduling_budget.yml")
# Stage 6's ModelRegistry (AlgorithmSpec entries) — read directly by
# ModelSelector at runtime, ML and deterministic algorithms registered the
# same way. NOT to be confused with MODEL_REGISTRY_PATH above (ml/model_
# registry.json — trained-model timestamps/metrics, a different concept
# from a different, earlier piece of work). See config/model_registry.yml's
# own header for the full rationale.
ALGORITHM_REGISTRY_CONFIG = _load_yaml(CONFIG_DIR / "model_registry.yml")

# ── Agent Config Accessors ────────────────────────────────────────────────────
# Agent 3 redesign, Phase 4 (plan "zany-giggling-crayon") — dropped
# TOOL_REGISTRY/EXECUTION_PLANS/DATASET_CTX, which read agent.yaml's
# "tools"/"execution_plans"/"dataset_context" sections: none of the three
# were read by any code even before this redesign (the old handlers
# hardcoded their own tool instances/intent routing directly), and the
# generic Stage 1-9 pipeline replaces the "execution_plans" concept
# entirely — see nodes/pipeline.py's _build_tools_used(). agent.yaml keeps
# those sections as human-readable documentation only.
AGENT_IDENTITY    = AGENT_CONFIG.get("agent", {})
AGENT_ROLE        = AGENT_CONFIG.get("role", {})
MEMORY_CONFIG     = AGENT_CONFIG.get("memory", {})
SYSTEM_PROMPT_CFG = AGENT_CONFIG.get("system_prompt", {})

# ── ML Config Accessors ───────────────────────────────────────────────────────
ML_SETTINGS         = ML_CONFIG.get("ml_settings", {})
ML_READINESS_THRESHOLD  = float(ML_SETTINGS.get("readiness_threshold", 75.0))
LLM_READINESS_THRESHOLD = float(ML_SETTINGS.get("llm_readiness_threshold", 75.0))
ML_FALLBACK_ENABLED = ML_SETTINGS.get("fallback_on_low_readiness", True)
ML_MODEL_SAVE_DIR   = BASE_DIR / ML_SETTINGS.get("model_save_dir", "ml/trained")

PROPHET_CFG    = ML_CONFIG.get("prophet", {})
LGBM_CFG       = ML_CONFIG.get("lightgbm", {})
ISO_FOREST_CFG = ML_CONFIG.get("isolation_forest", {})
XGBOOST_CFG    = ML_CONFIG.get("xgboost", {})
KMEANS_CFG     = ML_CONFIG.get("kmeans", {})
LLM_FALLBACK   = ML_CONFIG.get("llm_fallback", {})

# ── Business Rules Accessors ──────────────────────────────────────────────────
PREDEFINED_RULES  = RULES_CONFIG.get("predefined_rules", [])
KPI_THRESHOLDS    = RULES_CONFIG.get("thresholds", {})
VARIANCE_DRIVERS  = RULES_CONFIG.get("variance_drivers", {})
FLAG_DECODING     = RULES_CONFIG.get("flags", {})
NEW_RULES         = RULES_CONFIG.get("new_rules", [])
SUPPORTED_OPS     = RULES_CONFIG.get("supported_operations", [])

# ── Rules Files (JSON) ─────────────────────────────────────────────────────────
KPI_DEFINITIONS_PATH      = RULES_DIR / "kpi_definitions.json"
DRILL_DOWN_HIERARCHY_PATH = RULES_DIR / "drill_down_hierarchy.json"
BUSINESS_RULES_YAML_PATH  = CONFIG_DIR / "business_rules.yml"

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
# This project runs on Azure OpenAI, not Groq. AZURE_OPENAI_API_KEY/ENDPOINT
# normally come from the shared root .env (see this module's docstring); a
# service-local .env value always wins if both define the same key.
AZURE_OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_TEMPERATURE = 0.0
AZURE_OPENAI_MAX_TOKENS  = 1024

# ── ML Runtime Settings ───────────────────────────────────────────────────────
ANOMALY_CONTAMINATION = float(
    ISO_FOREST_CFG.get("hyperparameters", {}).get("contamination", 0.05)
)
FORECAST_PERIODS = int(
    PROPHET_CFG.get("forecast_horizons", {}).get("default_periods", 6)
)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ── Reporting ──────────────────────────────────────────────────────────────────
REPORTING_CURRENCY = "USD"
AMOUNT_UNIT        = "USD thousands"

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY_PATH = ML_DIR / "model_registry.json"
