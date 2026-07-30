"""Application-wide constants."""

# Pipeline version for traceability
PIPELINE_VERSION = "1.0.0"

# Confidence thresholds for secondary domain classification
SECONDARY_DOMAIN_CLASSIFIED_THRESHOLD = 0.75
SECONDARY_DOMAIN_NEEDS_REVIEW_THRESHOLD = 0.50

# Readiness thresholds
READINESS_READY_THRESHOLD = 80.0
READINESS_PARTIALLY_READY_THRESHOLD = 60.0

# Hierarchy thresholds
HIERARCHY_ACCEPTED_CONSISTENCY = 0.98
HIERARCHY_WARNING_CONSISTENCY = 0.90
HIERARCHY_MIN_MAPPING_COVERAGE = 0.90
IDENTIFIER_CARDINALITY_THRESHOLD = 0.98

# Maximum conflict samples in hierarchy edge results
MAX_CONFLICT_SAMPLES = 5

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
