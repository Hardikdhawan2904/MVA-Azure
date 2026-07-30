"""Configuration repository — loads and caches YAML domain/weight configs."""

from pathlib import Path
from typing import Any

import yaml

from app.core.exceptions import ConfigurationError

# This service's own root (app/repositories/../../.. -> Data-Profiling-Agent/).
# Used to resolve a relative config_dir against a fixed, known location
# instead of the process's current working directory.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigurationRepository:
    """Loads versioned YAML configuration files from disk."""

    def __init__(self, config_dir: str | Path = "config"):
        config_dir = Path(config_dir)
        # Every real call site passes the literal string "config" — that
        # only ever worked because start-all.ps1 always `cd`s into this
        # service's directory before launching uvicorn. Any other launch
        # method (Docker, systemd, CI) that doesn't set the working
        # directory to this exact folder would fail on the first request
        # with CONFIG_FILE_NOT_FOUND. Resolve relative paths against this
        # package's own location instead, so it works regardless of
        # launch directory. Absolute paths (as several tests already pass)
        # are used as-is, unaffected.
        self._config_dir = config_dir if config_dir.is_absolute() else _SERVICE_ROOT / config_dir
        self._cache: dict[str, Any] = {}

    def _load_yaml(self, relative_path: str) -> dict[str, Any]:
        """Load and cache a YAML file."""
        if relative_path in self._cache:
            return self._cache[relative_path]

        full_path = self._config_dir / relative_path
        if not full_path.exists():
            raise ConfigurationError(
                code="CONFIG_FILE_NOT_FOUND",
                message=f"Configuration file not found: {relative_path}",
                details={"path": str(full_path)},
            )

        with open(full_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._cache[relative_path] = data
        return data

    def get_domain_config(self, domain_name: str) -> dict[str, Any]:
        """Load domain configuration by name (case-insensitive file match)."""
        filename = f"domains/{domain_name.lower()}.yaml"
        return self._load_yaml(filename)

    def get_quality_weights(self) -> dict[str, Any]:
        """Load quality dimension weights."""
        return self._load_yaml("quality_weights.yaml")

    def get_readiness_weights(self) -> dict[str, Any]:
        """Load AI readiness weight profiles. Tested (tests/unit/test_config.py)
        and working, but currently unwired — ReadinessEngine's actual scoring
        weights are hardcoded in Python instead (see README's Configuration
        section). Not dead code, just not yet consumed by anything."""
        return self._load_yaml("readiness_weights.yaml")

    def get_hierarchy_thresholds(self) -> dict[str, Any]:
        """Load hierarchy validation thresholds."""
        return self._load_yaml("hierarchy_thresholds.yaml")

    def get_chart_policy(self) -> dict[str, Any]:
        """Load chart intelligence policy."""
        return self._load_yaml("chart_policy.yaml")

    def get_classification_weights(self) -> dict[str, Any]:
        """Load column data-category classification weights."""
        return self._load_yaml("classification_weights.yaml")

    def get_supported_primary_domains(self) -> list[str]:
        """Return list of supported primary domains from domain config files."""
        domains_dir = self._config_dir / "domains"
        if not domains_dir.exists():
            return []
        domains = []
        for f in domains_dir.glob("*.yaml"):
            data = self._load_yaml(f"domains/{f.name}")
            if "domain" in data:
                domains.append(data["domain"])
        return domains

    def get_secondary_domains_for(self, primary_domain: str) -> dict[str, Any]:
        """Return secondary domain definitions for a primary domain."""
        config = self.get_domain_config(primary_domain)
        return config.get("secondary_domains", {})

    def invalidate_cache(self) -> None:
        """Clear the configuration cache (useful for testing)."""
        self._cache.clear()
