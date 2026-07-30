"""Tests for deterministic type refinement."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import RefinedDataType
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


def _profile(profiler: ColumnProfiler, values: list, name: str = "col"):
    series = pd.Series(values)
    return profiler.profile_column(series, name, name)


class TestTypeRefiner:
    """Test deterministic type refinement logic."""

    def test_integer_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["1", "2", "3", "100", "999"])
        result = refiner.refine(profile)
        assert result == RefinedDataType.INTEGER

    def test_decimal_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["1.50", "2.75", "3.00", "100.99", "50.25"])
        result = refiner.refine(profile)
        assert result == RefinedDataType.DECIMAL

    def test_date_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"
        ])
        result = refiner.refine(profile)
        assert result == RefinedDataType.DATE

    def test_datetime_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "2024-01-01T10:30:00", "2024-02-15T14:45:00",
            "2024-03-20T08:00:00", "2024-04-10T16:20:00",
        ])
        result = refiner.refine(profile)
        assert result in (RefinedDataType.DATE, RefinedDataType.DATETIME)

    def test_boolean_true_false(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["true", "false", "true", "false", "true"] * 10)
        result = refiner.refine(profile)
        assert result == RefinedDataType.BOOLEAN

    def test_boolean_zero_one(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["0", "1", "1", "0", "1"] * 10)
        result = refiner.refine(profile)
        assert result == RefinedDataType.BOOLEAN

    def test_currency_code_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["USD", "EUR", "GBP", "JPY", "INR", "USD", "EUR"] * 5)
        result = refiner.refine(profile)
        assert result == RefinedDataType.CURRENCY_CODE

    def test_country_code_2_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["US", "GB", "DE", "FR", "JP", "IN", "AU"] * 5)
        result = refiner.refine(profile)
        assert result == RefinedDataType.COUNTRY_CODE

    def test_email_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "alice@example.com", "bob@test.org", "charlie@mail.co",
            "dave@company.io", "eve@domain.net",
        ])
        result = refiner.refine(profile)
        assert result == RefinedDataType.EMAIL

    def test_percentage_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["10", "25", "50", "75", "100", "0", "33"], "success_pct")
        result = refiner.refine(profile)
        assert result == RefinedDataType.PERCENTAGE

    def test_identifier_uuid(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        import uuid
        values = [str(uuid.uuid4()) for _ in range(50)]
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.IDENTIFIER

    def test_identifier_high_cardinality(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = [f"TXN-{i:06d}" for i in range(100)]
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.IDENTIFIER

    def test_categorical_low_cardinality(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = ["approved", "declined", "pending"] * 50
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.CATEGORICAL

    def test_text_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = [
            "This is a long description of the transaction",
            "Another detailed note about the payment processing",
            "Short note",
            "Yet another explanation of what happened during settlement",
            "Final remark about the authorization process",
        ] * 10 + ["Repeated text for variety"] * 10
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result in (RefinedDataType.TEXT, RefinedDataType.CATEGORICAL)

    def test_all_null_returns_unknown(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [None, None, None])
        result = refiner.refine(profile)
        assert result == RefinedDataType.UNKNOWN

    def test_low_cardinality_code_not_identifier(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A column like department_code with low cardinality should NOT be identifier."""
        values = ["DEPT01", "DEPT02", "DEPT03"] * 50
        profile = _profile(profiler, values, "department_code")
        result = refiner.refine(profile)
        # Should be categorical, not identifier
        assert result == RefinedDataType.CATEGORICAL

    def test_bare_digit_code_not_phone(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """An 8-digit bank identification number (or any bare, unformatted
        numeric code) must not be classified as PHONE — a real phone number
        represented as a string almost always has a separator or "+" prefix;
        a bare digit run is indistinguishable from any other numeric code."""
        values = ["51234567", "52345678", "53456789", "54567890", "55678901"] * 10
        profile = _profile(profiler, values, "BIN8")
        result = refiner.refine(profile)
        assert result != RefinedDataType.PHONE
        assert result == RefinedDataType.INTEGER

    def test_formatted_phone_still_detected(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A genuinely phone-shaped string (with separators) should still be
        detected — the fix narrows PHONE to require a separator/"+", it
        shouldn't eliminate real phone detection entirely."""
        values = ["+1-555-123-4567", "+1-555-987-6543", "(555) 234-5678", "555-345-6789"] * 5
        profile = _profile(profiler, values, "contact_phone")
        result = refiner.refine(profile)
        assert result == RefinedDataType.PHONE

    def test_bare_digit_phone_detected_via_name_hint(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A bare (unformatted) digit run of plausible phone length IS still
        detected as PHONE when the column name actually says so — the fix
        narrows false positives on codes, it shouldn't blanket-eliminate
        legitimate unformatted phone number columns."""
        values = [f"98765{i:05d}" for i in range(50)]
        profile = _profile(profiler, values, "customer_phone_number")
        result = refiner.refine(profile)
        assert result == RefinedDataType.PHONE

    def test_automobile_column_not_false_matched_as_phone(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """Regression guard: the phone name-hint match must be word-bounded,
        not a raw substring check — "automobile_id" contains "mobile" as a
        substring but has nothing to do with phone numbers."""
        values = [f"{20000000 + i}" for i in range(50)]
        profile = _profile(profiler, values, "automobile_id_number")
        result = refiner.refine(profile)
        assert result != RefinedDataType.PHONE

    def test_zero_heavy_column_with_fractional_tail_is_decimal(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A mostly-zero numeric column with a genuine fractional tail (e.g.
        estimated_decline_loss_usd, ~75% exact 0) must be DECIMAL, not
        INTEGER — the frequency-dominant "0" pattern must not shortcut past
        the real min/max check."""
        values = ["0"] * 75 + ["0.83", "4.44", "1.2", "2.5", "0.75"] * 5
        profile = _profile(profiler, values, "estimated_decline_loss_usd")
        result = refiner.refine(profile)
        assert result == RefinedDataType.DECIMAL

    def test_unique_long_commentary_not_identifier(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A free-text commentary column where every row happens to be a unique
        sentence should NOT be an identifier — high cardinality alone isn't
        enough when the values are long narrative text, not short codes."""
        values = [
            f"Total Operating Income of ${17000 + i}K was ${1000 + i}K below budget, "
            f"primarily driven by volume and rate variance in the reporting period."
            for i in range(60)
        ]
        profile = _profile(profiler, values, "commentary_headline")
        result = refiner.refine(profile)
        assert result != RefinedDataType.IDENTIFIER
