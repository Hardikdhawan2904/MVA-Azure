"""
File validation service.
Shared helper for extracting a file's extension — the rest of validation
(supported-extension check, dataframe structure, missing-filename check) now
lives in app/agents/schema_intelligence_agent/nodes/pipeline.py's validate_and_load node and the
upload_dataset route, since those need to raise/route within the graph's
control flow rather than as standalone pre-checks.
"""

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def get_file_extension(filename: str) -> str:
    """Extract and return the lowercase file extension."""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()
