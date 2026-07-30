"""Tests for configuration loading."""

import pytest

from app.core.config import Settings
from app.repositories.configuration_repository import ConfigurationRepository
from app.core.exceptions import ConfigurationError


class TestSettings:
    """Test application settings."""

    def test_default_settings(self):
        settings = Settings(
            DATABASE_URL="postgresql://x:x@localhost/test",
        )
        assert settings.max_upload_size_mb == 25
        assert settings.max_dataset_rows == 200_000
        assert settings.max_dataset_columns == 200
        assert settings.processing_timeout_seconds == 120
        assert settings.min_cube_group_size == 5

    def test_max_upload_size_bytes(self):
        settings = Settings(
            DATABASE_URL="postgresql://x:x@localhost/test",
            MAX_UPLOAD_SIZE_MB=10,
        )
        assert settings.max_upload_size_bytes == 10 * 1024 * 1024


class TestConfigurationRepository:
    """Test YAML configuration loading."""

    def test_load_payments_domain(self, config_repo: ConfigurationRepository):
        config = config_repo.get_domain_config("Payments")
        assert config["domain"] == "Payments"
        assert "Authorization" in config["secondary_domains"]
        assert "Clearing" in config["secondary_domains"]
        assert "Settlement" in config["secondary_domains"]
        assert "Fraud" in config["secondary_domains"]

    def test_load_customer_domain(self, config_repo: ConfigurationRepository):
        config = config_repo.get_domain_config("Customer")
        assert config["domain"] == "Customer"
        assert "CRM" in config["secondary_domains"]

    def test_load_hr_domain(self, config_repo: ConfigurationRepository):
        config = config_repo.get_domain_config("HR")
        assert config["domain"] == "HR"
        assert "Employee" in config["secondary_domains"]

    def test_load_finance_domain(self, config_repo: ConfigurationRepository):
        config = config_repo.get_domain_config("Finance")
        assert config["domain"] == "Finance"
        assert "Revenue" in config["secondary_domains"]

    def test_missing_config_raises_error(self, config_repo: ConfigurationRepository):
        with pytest.raises(ConfigurationError) as exc_info:
            config_repo.get_domain_config("NonExistent")
        assert exc_info.value.code == "CONFIG_FILE_NOT_FOUND"

    def test_supported_primary_domains(self, config_repo: ConfigurationRepository):
        domains = config_repo.get_supported_primary_domains()
        assert "Payments" in domains
        assert "Customer" in domains
        assert "HR" in domains
        assert "Finance" in domains

    def test_secondary_domains_for_payments(self, config_repo: ConfigurationRepository):
        secondary = config_repo.get_secondary_domains_for("Payments")
        assert "Authorization" in secondary
        assert "keywords" in secondary["Authorization"]

    def test_quality_weights(self, config_repo: ConfigurationRepository):
        weights = config_repo.get_quality_weights()
        assert weights["version"] == 1
        assert "completeness" in weights["weights"]
        total = sum(weights["weights"].values())
        assert abs(total - 1.0) < 0.001

    def test_readiness_weights(self, config_repo: ConfigurationRepository):
        weights = config_repo.get_readiness_weights()
        assert "analytics" in weights
        assert "ml" in weights
        assert "llm" in weights

    def test_hierarchy_thresholds(self, config_repo: ConfigurationRepository):
        thresholds = config_repo.get_hierarchy_thresholds()
        assert thresholds["accepted_consistency"] == 0.98
        assert thresholds["warning_consistency"] == 0.90
        assert thresholds["min_mapping_coverage"] == 0.90

    def test_chart_policy(self, config_repo: ConfigurationRepository):
        policy = config_repo.get_chart_policy()
        assert policy["max_charts"] == 5
        assert "bar" in policy["chart_type_requirements"]

    def test_cache_invalidation(self, config_repo: ConfigurationRepository):
        config_repo.get_domain_config("Payments")
        assert "domains/payments.yaml" in config_repo._cache
        config_repo.invalidate_cache()
        assert len(config_repo._cache) == 0
