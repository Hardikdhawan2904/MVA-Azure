"""Shared enumerations used across the application."""

from enum import Enum


class PrimaryDomain(str, Enum):
    """Supported primary domains for v1."""
    PAYMENTS = "Payments"
    CUSTOMER = "Customer"
    HR = "HR"
    FINANCE = "Finance"


class RunStatus(str, Enum):
    """Processing run statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    """Status for individual pipeline stages."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NEEDS_REVIEW = "needs_review"
    NOT_ASSESSABLE = "not_assessable"
    FAILED = "failed"


class SecondaryDomainStatus(str, Enum):
    """Secondary domain classification status."""
    CLASSIFIED = "classified"
    NEEDS_REVIEW = "needs_review"
    UNRESOLVED = "unresolved"


class RefinedDataType(str, Enum):
    """Refined physical data types beyond pandas dtype."""
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IDENTIFIER = "identifier"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY_CODE = "currency_code"
    COUNTRY_CODE = "country_code"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    """Semantic column roles."""
    METRIC = "metric"
    DIMENSION = "dimension"
    IDENTIFIER = "identifier"
    TEMPORAL_DIMENSION = "temporal_dimension"
    TEXT_FIELD = "text_field"
    FLAG = "flag"
    UNKNOWN = "unknown"


class DataCategory(str, Enum):
    """Allowed data categories for column classification."""
    DEMOGRAPHIC = "Demographic Data"
    TRANSACTION = "Transaction Data"
    FINANCIAL = "Financial Data"
    BEHAVIORAL = "Behavioral Data"
    OPERATIONAL = "Operational Data"
    INTERACTION = "Interaction Data"
    MASTER = "Master Data"
    TIME_SERIES = "Time Series Data"
    GEOGRAPHIC = "Geographic Data"
    RISK = "Risk Data"
    CUSTOMER_EXPERIENCE = "Customer Experience Data"


class QualityDimension(str, Enum):
    """Data quality assessment dimensions."""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONFORMITY = "conformity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    ACCURACY = "accuracy"
    BUSINESS_RULE_COMPLIANCE = "business_rule_compliance"
    SEMANTIC_QUALITY = "semantic_quality"


class QualityStatus(str, Enum):
    """Quality dimension assessment status."""
    ASSESSED = "assessed"
    NOT_ASSESSABLE = "not_assessable"
    FAILED = "failed"


class ReadinessType(str, Enum):
    """AI readiness assessment types."""
    ANALYTICS = "analytics_readiness"
    ML = "ml_readiness"
    LLM = "llm_readiness"
    OVERALL = "overall_ai_readiness"


class ReadinessStatus(str, Enum):
    """Readiness status thresholds."""
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"


class ChartType(str, Enum):
    """Allowed chart types for v1."""
    BAR = "bar"
    LINE = "line"
    STACKED_BAR = "stacked_bar"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    KPI = "kpi"


class ChartCategory(str, Enum):
    """Chart categories."""
    BUSINESS = "business"
    PROFILING = "profiling"
    QUALITY = "quality"


class HierarchyEdgeStatus(str, Enum):
    """Functional dependency validation status for hierarchy edges."""
    ACCEPTED = "accepted"
    RETAINED_WITH_WARNING = "retained_with_warning"
    REJECTED = "rejected"


class HierarchyChainStatus(str, Enum):
    """Overall hierarchy chain status."""
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class RuleType(str, Enum):
    """Supported business rule types."""
    NON_NULL = "non_null"
    EXPECTED_UNIQUE = "expected_unique"
    REGEX_MATCH = "regex_match"
    ALLOWED_VALUES = "allowed_values"
    NUMERIC_RANGE = "numeric_range"
    DATE_RANGE = "date_range"
    COLUMN_COMPARISON = "column_comparison"
    CONDITIONAL_REQUIRED = "conditional_required"
    CROSS_FIELD_EQUALITY = "cross_field_equality"
    CROSS_FIELD_INEQUALITY = "cross_field_inequality"


class RuleSuggestionStatus(str, Enum):
    """Rule suggestion lifecycle statuses."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class RuleSource(str, Enum):
    """Source of a business rule."""
    DOMAIN_CONFIGURATION = "domain_configuration"
    REQUEST = "request"
    LLM_SUGGESTION = "llm_suggestion"
    APPROVED_SUGGESTION = "approved_suggestion"


class SchemaIntelligenceDecision(str, Enum):
    """Schema Intelligence decisions for semantic candidates."""
    CONFIRMED = "confirmed"
    OVERRIDDEN = "overridden"
    UNRESOLVED = "unresolved"


class FileType(str, Enum):
    """Supported file types."""
    CSV = "csv"
    XLSX = "xlsx"
