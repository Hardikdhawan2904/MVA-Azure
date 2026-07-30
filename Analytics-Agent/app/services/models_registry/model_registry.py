"""app/services/models_registry/model_registry.py — Stage 6 of the Agent 3
redesign (plan "zany-giggling-crayon"): ModelRegistry.

Loads config/model_registry.yml into AlgorithmSpec entries — read directly
by ModelSelector at runtime (unlike ml_config.yml, which only ever fed
train.py, not query-time code). Adding a new algorithm is one new YAML
entry + one implementation class; zero changes to ModelSelector's logic.
"""

from __future__ import annotations

from app.config import ALGORITHM_REGISTRY_CONFIG
from app.services.models_registry.algorithm_spec import AlgorithmSpec


class ModelRegistry:
    def __init__(self, config: dict | None = None):
        raw = config if config is not None else ALGORITHM_REGISTRY_CONFIG
        self._algorithms = [AlgorithmSpec.from_dict(a) for a in raw.get("algorithms", [])]

    def all(self) -> list[AlgorithmSpec]:
        return list(self._algorithms)

    def for_purpose(self, purpose: str, enabled_only: bool = True) -> list[AlgorithmSpec]:
        specs = [a for a in self._algorithms if a.purpose == purpose]
        if enabled_only:
            specs = [a for a in specs if a.enabled]
        return specs

    def ml_algorithms_for(self, purpose: str) -> list[AlgorithmSpec]:
        return [a for a in self.for_purpose(purpose) if a.requirements.requires_ml]

    def deterministic_algorithms_for(self, purpose: str) -> list[AlgorithmSpec]:
        return [a for a in self.for_purpose(purpose) if not a.requirements.requires_ml]
