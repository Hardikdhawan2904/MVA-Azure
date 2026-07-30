"""
tools/knowledge_update_tool.py — Tool 6: Rule Engine Updater

If a KPI or business concept is missing from the Rule Engine:
1. Generate a candidate business rule using Azure OpenAI.
2. Validate the candidate (structure + plausibility).
3. If valid, update the Rule Engine.
4. Continue the analysis.

Never updates the Rule Engine without validation.
"""

import json
import logging

import httpx

from app.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """You are a business rules generator for an Enterprise Insurance Analytics Platform.

A new KPI has been requested that is not in our Rule Engine.
Generate a valid KPI definition as a JSON object.

The JSON must contain:
- "label": Full KPI name (string)
- "formula": Plain-language formula description (string)
- "unit": e.g. "%" or "USD thousands" or "ratio" (string)
- "category": One of: "premium", "claims", "ratio", "policy", "expense", "profitability", "reserve" (string)
- "higher_is_better": true or false (boolean)
- "dependencies": List of related KPI keys (list of strings, can be empty)
- "thresholds": Optional dict with "healthy", "warning", "critical" values

Return ONLY the JSON object. No explanation. No markdown."""


class KnowledgeUpdateTool:
    """
    Generates and validates new KPI definitions when the Rule Engine
    does not contain a requested metric.
    """

    def __init__(self, rule_engine):
        self.rule_engine = rule_engine
        self._client = None
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT:
            self._client = httpx.Client(timeout=30.0)
            logger.info("KnowledgeUpdateTool: Azure OpenAI client ready")
        else:
            logger.warning("Azure OpenAI not configured — knowledge generation disabled")

    def update(self, kpi_name: str) -> dict:
        """
        Main entry point. Given an unknown KPI name:
        1. Generate candidate definition.
        2. Validate.
        3. Update Rule Engine if valid.

        Returns:
            {
              "success": bool,
              "kpi_key": str,
              "definition": dict | None,
              "reason": str,
            }
        """
        logger.info(f"KnowledgeUpdateTool: requested KPI '{kpi_name}'")

        # Step 1: Generate candidate
        candidate = self._generate_candidate(kpi_name)
        if not candidate:
            return {
                "success": False,
                "kpi_key": kpi_name,
                "definition": None,
                "reason": "Could not generate candidate — Azure OpenAI unavailable or returned invalid JSON",
            }

        # Step 2: Validate
        is_valid, reason = self._validate(candidate)
        if not is_valid:
            logger.warning(f"Candidate for '{kpi_name}' failed validation: {reason}")
            return {
                "success": False,
                "kpi_key": kpi_name,
                "definition": candidate,
                "reason": f"Validation failed: {reason}",
            }

        # Step 3: Persist to Rule Engine
        key = kpi_name.lower().replace(" ", "_").replace("-", "_")
        added = self.rule_engine.add_kpi(key, candidate)
        if not added:
            return {
                "success": False,
                "kpi_key": key,
                "definition": candidate,
                "reason": "KPI already exists in Rule Engine",
            }

        logger.info(f"Rule Engine updated with new KPI: '{key}'")
        return {
            "success": True,
            "kpi_key": key,
            "definition": candidate,
            "reason": "Validated and added to Rule Engine",
        }

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_candidate(self, kpi_name: str) -> dict | None:
        if not self._client:
            logger.warning("Azure OpenAI unavailable — using minimal stub for known insurance KPIs")
            return self._static_fallback(kpi_name)

        try:
            response = self._client.post(
                f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/chat/completions",
                headers={
                    "api-key": AZURE_OPENAI_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "model": AZURE_OPENAI_DEPLOYMENT,
                    "messages": [
                        {"role": "system", "content": _GENERATION_PROMPT},
                        {"role": "user",   "content": f"Generate a definition for KPI: {kpi_name}"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            candidate = json.loads(raw)
            logger.info(f"Generated candidate for '{kpi_name}': {list(candidate.keys())}")
            return candidate
        except Exception as e:
            logger.error(f"Generation error for '{kpi_name}': {e}")
            return None

    def _static_fallback(self, kpi_name: str) -> dict | None:
        """
        Hard-coded fallback for common insurance KPIs not yet in the engine.
        Used when Azure OpenAI is unavailable.
        """
        known = {
            "net_promoter_score": {
                "label": "Net Promoter Score (NPS)",
                "formula": "% Promoters - % Detractors",
                "unit": "score",
                "category": "customer",
                "higher_is_better": True,
                "dependencies": [],
            },
            "ibnr_ratio": {
                "label": "IBNR to Reserve Ratio",
                "formula": "IBNR Reserve / (IBNR Reserve + Outstanding Claim Reserve) * 100",
                "unit": "%",
                "category": "reserve",
                "higher_is_better": False,
                "dependencies": ["ibnr_reserve", "outstanding_claim_reserve"],
            },
        }
        key = kpi_name.lower().replace(" ", "_")
        return known.get(key)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, candidate: dict) -> tuple[bool, str]:
        """
        Validate that a candidate KPI definition is structurally sound using Pydantic.
        """
        from app.services.schemas import KPIDefinition
        try:
            ALLOWED_CATEGORIES = {
                "premium", "claims", "ratio", "policy",
                "expense", "profitability", "reserve", "customer",
            }
            if candidate.get("category") not in ALLOWED_CATEGORIES:
                return False, f"category must be one of {ALLOWED_CATEGORIES}, got '{candidate.get('category')}'"

            # Check if all required fields are present (since Pydantic fields without default are required)
            KPIDefinition.model_validate(candidate)
            return True, "OK"
        except Exception as e:
            return False, str(e)
