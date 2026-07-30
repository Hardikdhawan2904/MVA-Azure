"""Deterministic type refinement — refines pandas dtype to semantic physical types."""

import re

from app.core.enums import RefinedDataType
from app.services.profiling.column_profiler import ColumnProfileResult

# Known ISO currency codes (subset for demo — production would use full ISO 4217)
_CURRENCY_CODES = {
    "USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF", "CNY", "HKD",
    "NZD", "SEK", "KRW", "SGD", "NOK", "MXN", "BRL", "ZAR", "DKK", "PLN",
    "THB", "IDR", "HUF", "CZK", "ILS", "CLP", "PHP", "AED", "COP", "SAR",
    "MYR", "RON", "TRY", "TWD", "ARS", "VND", "EGP", "NGN", "PKR", "BDT",
}

# ISO 3166-1 alpha-2 country codes (subset)
_COUNTRY_CODES_2 = {
    "US", "GB", "DE", "FR", "JP", "IN", "AU", "CA", "CH", "CN",
    "BR", "MX", "ZA", "KR", "SG", "NZ", "SE", "NO", "DK", "NL",
    "IT", "ES", "PT", "BE", "AT", "IE", "FI", "PL", "CZ", "HU",
    "RO", "GR", "TR", "RU", "AE", "SA", "EG", "NG", "KE", "IL",
}

# ISO 3166-1 alpha-3 country codes (subset)
_COUNTRY_CODES_3 = {
    "USA", "GBR", "DEU", "FRA", "JPN", "IND", "AUS", "CAN", "CHE", "CHN",
    "BRA", "MEX", "ZAF", "KOR", "SGP", "NZL", "SWE", "NOR", "DNK", "NLD",
}

_EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
# Requires an actual separator or leading "+" — a bare unbroken digit run
# (e.g. an 8-digit BIN/account code) is indistinguishable from any other
# numeric identifier and must not match here. Bare/unformatted real phone
# numbers are still recoverable via _is_bare_digit_phone_by_name below.
_PHONE_PATTERN = re.compile(r"^(?=.*[\s\-\(\)+])\+?\d[\d\s\-\(\)]{7,}$")
# Word-boundary (underscore-delimited) match, not raw substring — otherwise
# e.g. "automobile_type" would false-match on "mobile".
_PHONE_NAME_HINT_PATTERN = re.compile(
    r"(?:^|_)(phone|mobile|cell|telephone|contact_?no|contact_?number|fax)(?:_|$)", re.IGNORECASE
)


class TypeRefiner:
    """Refines column types from pandas object dtype to semantic physical types."""

    def refine(self, profile: ColumnProfileResult) -> RefinedDataType:
        """
        Determine the refined physical data type for a column.

        Uses parse ratios, patterns, cardinality, and value analysis.
        Does NOT use LLM — purely deterministic.
        """
        # If all null, unknown
        if profile.non_null_count == 0:
            return RefinedDataType.UNKNOWN

        # Check boolean first (before numeric, since 0/1 also parses as numeric)
        if self._is_boolean(profile):
            return RefinedDataType.BOOLEAN

        # Check datetime
        if self._is_datetime(profile):
            if self._is_date_only(profile):
                return RefinedDataType.DATE
            return RefinedDataType.DATETIME

        # Check email
        if self._is_email(profile):
            return RefinedDataType.EMAIL

        # Check phone
        if self._is_phone(profile):
            return RefinedDataType.PHONE

        # Check currency code
        if self._is_currency_code(profile):
            return RefinedDataType.CURRENCY_CODE

        # Check country code
        if self._is_country_code(profile):
            return RefinedDataType.COUNTRY_CODE

        # Check percentage
        if self._is_percentage(profile):
            return RefinedDataType.PERCENTAGE

        # Check numeric types
        if profile.numeric_parse_ratio >= 0.90:
            if self._is_integer(profile):
                return RefinedDataType.INTEGER
            return RefinedDataType.DECIMAL

        # Check identifier (near-unique string)
        if self._is_identifier(profile):
            return RefinedDataType.IDENTIFIER

        # Check categorical (low cardinality)
        if self._is_categorical(profile):
            return RefinedDataType.CATEGORICAL

        # Default to text
        if profile.non_null_count > 0:
            return RefinedDataType.TEXT

        return RefinedDataType.UNKNOWN

    def _is_boolean(self, profile: ColumnProfileResult) -> bool:
        """Column is boolean if high boolean parse ratio and very low cardinality."""
        if profile.boolean_parse_ratio >= 0.95 and profile.distinct_count <= 3:
            return True
        # Also check for numeric 0/1 only pattern
        if (
            profile.numeric_parse_ratio >= 0.95
            and profile.distinct_count <= 2
            and profile.min_value is not None
            and profile.max_value is not None
            and profile.min_value >= 0
            and profile.max_value <= 1
        ):
            return True
        return False

    def _is_datetime(self, profile: ColumnProfileResult) -> bool:
        """Column is datetime if high datetime parse ratio."""
        return profile.datetime_parse_ratio >= 0.90

    def _is_date_only(self, profile: ColumnProfileResult) -> bool:
        """Check if datetime values are date-only (no time component)."""
        return "YYYY-MM-DD" in profile.dominant_patterns or "DATE_SLASH" in profile.dominant_patterns

    def _is_email(self, profile: ColumnProfileResult) -> bool:
        """Check if values match email pattern."""
        if "EMAIL" in profile.dominant_patterns:
            return True
        if profile.non_null_count == 0:
            return False
        # Check representative values
        email_count = sum(
            1 for v in profile.representative_values
            if isinstance(v, str) and _EMAIL_PATTERN.match(v)
        )
        return email_count / max(len(profile.representative_values), 1) >= 0.8

    def _is_phone(self, profile: ColumnProfileResult) -> bool:
        """Check if values match phone pattern.

        A formatted number (separator or "+" present) is recognized from
        shape alone via _PHONE_PATTERN. A bare, unformatted digit run is
        fundamentally ambiguous with any other numeric code (BIN, account
        number, zip, etc.) — it's only classified as PHONE when the column
        name itself says so, same disambiguation pattern _is_percentage()
        already uses for 0-100 ranges.
        """
        if "PHONE" in profile.dominant_patterns:
            return True
        if profile.non_null_count == 0:
            return False
        phone_count = sum(
            1 for v in profile.representative_values
            if isinstance(v, str) and _PHONE_PATTERN.match(v)
        )
        if phone_count / max(len(profile.representative_values), 1) >= 0.8:
            return True
        return self._is_bare_digit_phone_by_name(profile)

    def _is_bare_digit_phone_by_name(self, profile: ColumnProfileResult) -> bool:
        if not _PHONE_NAME_HINT_PATTERN.search(profile.normalized_key):
            return False
        if profile.numeric_parse_ratio < 0.90:
            return False
        if profile.min_string_length is None or profile.max_string_length is None:
            return False
        return 7 <= profile.min_string_length and profile.max_string_length <= 15

    def _is_currency_code(self, profile: ColumnProfileResult) -> bool:
        """Check if values are ISO currency codes."""
        if profile.distinct_count > 50:
            return False
        if "THREE_LETTER_CODE" not in profile.dominant_patterns:
            return False
        # Check representative values against known codes
        match_count = sum(
            1 for v in profile.representative_values
            if isinstance(v, str) and v.upper() in _CURRENCY_CODES
        )
        return match_count / max(len(profile.representative_values), 1) >= 0.7

    def _is_country_code(self, profile: ColumnProfileResult) -> bool:
        """Check if values are ISO country codes."""
        if profile.distinct_count > 250:
            return False
        # Two-letter codes
        if "TWO_LETTER_CODE" in profile.dominant_patterns:
            match_count = sum(
                1 for v in profile.representative_values
                if isinstance(v, str) and v.upper() in _COUNTRY_CODES_2
            )
            if match_count / max(len(profile.representative_values), 1) >= 0.7:
                return True
        # Three-letter codes
        if "THREE_LETTER_CODE" in profile.dominant_patterns:
            match_count = sum(
                1 for v in profile.representative_values
                if isinstance(v, str) and v.upper() in _COUNTRY_CODES_3
            )
            if match_count / max(len(profile.representative_values), 1) >= 0.7:
                return True
        return False

    def _is_percentage(self, profile: ColumnProfileResult) -> bool:
        """Check if numeric values are percentages (0-100 range with hints)."""
        if profile.numeric_parse_ratio < 0.90:
            return False
        if profile.min_value is None or profile.max_value is None:
            return False
        # Values constrained to 0-100
        if profile.min_value >= 0 and profile.max_value <= 100:
            # Check name hints
            name_lower = profile.normalized_key.lower()
            pct_hints = ("pct", "percent", "ratio", "rate", "proportion")
            if any(h in name_lower for h in pct_hints):
                return True
        return False

    def _is_integer(self, profile: ColumnProfileResult) -> bool:
        """Check if numeric values are integers (no decimals).

        Prefers all_integer_valued — checked against every parsed value in
        the column, not a sample — over dominant-pattern/min-max heuristics,
        which are frequency- or extremes-based and can be fooled by a
        zero-heavy column (mode "0" matching the INTEGER pattern) that hides
        a genuine fractional tail elsewhere in the distribution.
        """
        if profile.all_integer_valued is not None:
            return profile.all_integer_valued

        # Fallback for callers that construct a ColumnProfileResult without
        # all_integer_valued (e.g. hand-built test fixtures).
        if "DECIMAL_2DP" in profile.dominant_patterns:
            return False
        if profile.min_value is not None and profile.max_value is not None:
            if profile.min_value == int(profile.min_value) and profile.max_value == int(profile.max_value):
                for v in profile.sample_values:
                    try:
                        f = float(v)
                        if f != int(f):
                            return False
                    except (ValueError, TypeError):
                        pass
                return True
            return False
        return "INTEGER" in profile.dominant_patterns

    def _is_identifier(self, profile: ColumnProfileResult) -> bool:
        """Check if column is likely an identifier (near-unique)."""
        if "UUID" in profile.dominant_patterns:
            return True
        # Long free-text values (commentary/narrative fields) are never real
        # identifiers even when every row happens to be unique — a genuine
        # ID/code is short. Guard the cardinality check with a length check
        # (same 40-char threshold used for free-text detection elsewhere).
        if profile.avg_string_length is not None and profile.avg_string_length > 40:
            return False
        if profile.cardinality_ratio >= 0.98 and profile.non_null_count > 10:
            return True
        return False

    def _is_categorical(self, profile: ColumnProfileResult) -> bool:
        """Check if column is categorical (low cardinality)."""
        if profile.non_null_count == 0:
            return False
        # Low distinct count relative to rows
        if profile.cardinality_ratio <= 0.05 and profile.distinct_count <= 50:
            return True
        if profile.distinct_count <= 20 and profile.non_null_count >= 50:
            return True
        return False
