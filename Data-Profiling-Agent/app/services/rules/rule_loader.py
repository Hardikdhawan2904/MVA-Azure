"""Rule loader — loads configured and request-level business rules."""

from typing import Any

from app.core.enums import RuleType, RuleSource
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository

logger = get_logger(__name__)


class RuleDefinition:
    """A loaded business rule ready for execution."""

    def __init__(
        self,
        rule_key: str,
        rule_type: RuleType,
        source: RuleSource,
        domain: str,
        secondary_domain: str | None,
        parameters: dict[str, Any],
        severity: str = "medium",
        active: bool = True,
    ):
        self.rule_key = rule_key
        self.rule_type = rule_type
        self.source = source
        self.domain = domain
        self.secondary_domain = secondary_domain
        self.parameters = parameters
        self.severity = severity
        self.active = active

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_key": self.rule_key,
            "rule_type": self.rule_type.value,
            "source": self.source.value,
            "domain": self.domain,
            "secondary_domain": self.secondary_domain,
            "parameters": self.parameters,
            "severity": self.severity,
            "active": self.active,
        }


class RuleLoader:
    """Loads business rules from configuration and request-level inputs."""

    def __init__(self, config_repo: ConfigurationRepository):
        self._config_repo = config_repo

    def load_domain_rules(
        self, primary_domain: str, secondary_domain: str | None = None
    ) -> list[RuleDefinition]:
        """Load active rules from domain configuration."""
        domain_config = self._config_repo.get_domain_config(primary_domain)
        raw_rules = domain_config.get("business_rules", [])

        rules: list[RuleDefinition] = []
        for raw in raw_rules:
            rule_type = self._parse_rule_type(raw.get("type", ""))
            if rule_type is None:
                logger.warning("invalid_rule_type", rule_key=raw.get("rule_key"), type=raw.get("type"))
                continue

            # Filter by secondary domain if specified
            rule_secondary = raw.get("secondary_domains", [])
            if secondary_domain and rule_secondary and secondary_domain not in rule_secondary:
                continue

            rules.append(RuleDefinition(
                rule_key=raw.get("rule_key", "unknown"),
                rule_type=rule_type,
                source=RuleSource.DOMAIN_CONFIGURATION,
                domain=primary_domain,
                secondary_domain=secondary_domain,
                parameters=self._extract_parameters(raw, rule_type),
                severity=raw.get("severity", "medium"),
                active=True,
            ))

        return rules

    def load_request_rules(
        self, request_rules: list[dict[str, Any]], primary_domain: str, source: RuleSource = RuleSource.REQUEST
    ) -> list[RuleDefinition]:
        """Load additional rules from the request payload (or from approved suggestions, via `source`)."""
        rules: list[RuleDefinition] = []
        for raw in request_rules:
            rule_type = self._parse_rule_type(raw.get("type", ""))
            if rule_type is None:
                continue

            rules.append(RuleDefinition(
                rule_key=raw.get("rule_key", f"request_rule_{len(rules)}"),
                rule_type=rule_type,
                source=source,
                domain=primary_domain,
                secondary_domain=None,
                parameters=self._extract_parameters(raw, rule_type),
                severity=raw.get("severity", "medium"),
                active=True,
            ))

        return rules

    def _parse_rule_type(self, type_str: str) -> RuleType | None:
        """Parse a rule type string to enum."""
        try:
            return RuleType(type_str)
        except ValueError:
            return None

    def _extract_parameters(self, raw: dict[str, Any], rule_type: RuleType) -> dict[str, Any]:
        """Extract rule-type-specific parameters."""
        params: dict[str, Any] = {}

        if rule_type == RuleType.NON_NULL:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")

        elif rule_type == RuleType.EXPECTED_UNIQUE:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")

        elif rule_type == RuleType.REGEX_MATCH:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")
            params["pattern"] = raw.get("pattern", "")

        elif rule_type == RuleType.ALLOWED_VALUES:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")
            params["values"] = raw.get("values", [])

        elif rule_type == RuleType.NUMERIC_RANGE:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")
            params["min_value"] = raw.get("min_value")
            params["max_value"] = raw.get("max_value")
            params["inclusive_min"] = raw.get("inclusive_min", True)
            params["inclusive_max"] = raw.get("inclusive_max", True)

        elif rule_type == RuleType.DATE_RANGE:
            params["target_column"] = raw.get("target_column")
            params["target_role"] = raw.get("target_role")
            params["min_date"] = raw.get("min_date")
            params["max_date"] = raw.get("max_date")

        elif rule_type == RuleType.COLUMN_COMPARISON:
            params["left_column"] = raw.get("left_column")
            params["left_role"] = raw.get("left_role")
            params["operator"] = raw.get("operator", ">=")
            params["right_column"] = raw.get("right_column")
            params["right_role"] = raw.get("right_role")

        elif rule_type in (RuleType.CROSS_FIELD_EQUALITY, RuleType.CROSS_FIELD_INEQUALITY):
            params["left_column"] = raw.get("left_column")
            params["right_column"] = raw.get("right_column")

        elif rule_type == RuleType.CONDITIONAL_REQUIRED:
            params["condition_column"] = raw.get("condition_column")
            params["condition_value"] = raw.get("condition_value")
            params["required_column"] = raw.get("required_column")

        return params
