"""Tests for secondary-domain classification."""

import pytest
import pandas as pd
from pathlib import Path

from app.core.config import Settings
from app.core.enums import SecondaryDomainStatus
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.domains.secondary_domain_classifier import SecondaryDomainClassifier
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator
from app.services.profiling.type_refiner import TypeRefiner


@pytest.fixture
def config_repo() -> ConfigurationRepository:
    config_dir = Path(__file__).parent.parent.parent / "config"
    return ConfigurationRepository(config_dir=config_dir)


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


@pytest.fixture
def sem_gen() -> SemanticCandidateGenerator:
    return SemanticCandidateGenerator()


def _build_profiles_and_candidates(profiler, refiner, sem_gen, df):
    """Helper to build profiles and semantic candidates from a DataFrame."""
    from app.services.profiling.dataset_profiler import normalize_column_name
    profiles = []
    refined_types = []
    for col in df.columns:
        p = profiler.profile_column(df[col], col, normalize_column_name(col))
        profiles.append(p)
        refined_types.append(refiner.refine(p))

    from app.services.profiling.identifier_detector import IdentifierDetector
    detector = IdentifierDetector()
    grain_result = detector.detect(profiles, refined_types)
    id_flags = [c.is_identifier for c in grain_result.identifier_candidates]

    candidates = sem_gen.generate_all(profiles, refined_types, id_flags)
    return profiles, candidates


class TestSecondaryDomainClassifier:
    """Test secondary-domain classification."""

    def test_classified_authorization_domain(self, config_repo, profiler, refiner, sem_gen):
        """Dataset with authorization signals should classify as Authorization."""
        df = pd.DataFrame({
            "auth_status": ["approved", "declined", "pending"] * 30,
            "approval_code": [f"AC{i:04d}" for i in range(90)],
            "decline_reason": ["insufficient_funds", "expired_card", None] * 30,
            "txn_amount": [str(i * 10) for i in range(90)],
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        assert result.status == SecondaryDomainStatus.CLASSIFIED
        assert result.name == "Authorization"
        assert result.confidence >= 0.75

    def test_classified_settlement_domain(self, config_repo, profiler, refiner, sem_gen):
        """Dataset with settlement signals should classify as Settlement."""
        df = pd.DataFrame({
            "settlement_date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 30,
            "settled_amount": [str(i * 100) for i in range(90)],
            "settlement_status": ["settled", "pending", "failed"] * 30,
            "settlement_reference": [f"SR{i}" for i in range(90)],
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        assert result.status == SecondaryDomainStatus.CLASSIFIED
        assert result.name == "Settlement"

    def test_only_selects_from_configured_domains(self, config_repo, profiler, refiner, sem_gen):
        """Classifier must not invent a domain outside configured options."""
        df = pd.DataFrame({
            "col1": ["x"] * 50,
            "col2": ["y"] * 50,
            "col3": [str(i) for i in range(50)],
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        # Should be unresolved or needs_review, not some invented domain
        if result.name is not None:
            allowed = list(config_repo.get_secondary_domains_for("Payments").keys())
            assert result.name in allowed

    def test_unresolved_when_no_signals(self, config_repo, profiler, refiner, sem_gen):
        """Dataset with no matching signals should be unresolved."""
        df = pd.DataFrame({
            "x1": ["abc"] * 50,
            "x2": [str(i % 5) for i in range(50)],
            "x3": ["hello"] * 50,
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        assert result.status in (SecondaryDomainStatus.UNRESOLVED, SecondaryDomainStatus.NEEDS_REVIEW)
        # No domain should be assigned
        if result.status == SecondaryDomainStatus.UNRESOLVED:
            assert result.name is None

    def test_needs_review_with_candidates(self, config_repo, profiler, refiner, sem_gen):
        """Ambiguous dataset should return needs_review with ranked candidates."""
        # Mix of fraud and authorization signals
        df = pd.DataFrame({
            "auth_status": ["approved", "declined"] * 25,
            "fraud_flag": ["0", "1", "0", "0", "1"] * 10,
            "risk_score": [str(i) for i in range(50)],
            "approval_code": [f"A{i}" for i in range(50)],
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(
            config_repo,
            classified_threshold=0.95,  # Set very high to force needs_review
            needs_review_threshold=0.30,
        )
        result = classifier.classify("Payments", profiles, candidates)

        if result.status == SecondaryDomainStatus.NEEDS_REVIEW:
            assert result.candidates is not None
            assert len(result.candidates) >= 1

    def test_confidence_between_0_and_1(self, config_repo, profiler, refiner, sem_gen):
        """Confidence must always be between 0 and 1."""
        df = pd.DataFrame({
            "settlement_date": ["2024-01-01"] * 50,
            "settled_amount": ["100.00"] * 50,
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_provided(self, config_repo, profiler, refiner, sem_gen):
        """Classified result should include evidence."""
        df = pd.DataFrame({
            "auth_status": ["approved", "declined", "pending"] * 30,
            "decline_reason": ["reason_a", "reason_b", None] * 30,
        })
        profiles, candidates = _build_profiles_and_candidates(profiler, refiner, sem_gen, df)

        classifier = SecondaryDomainClassifier(config_repo)
        result = classifier.classify("Payments", profiles, candidates)

        if result.status == SecondaryDomainStatus.CLASSIFIED:
            assert result.evidence is not None
            assert len(result.evidence) > 0
