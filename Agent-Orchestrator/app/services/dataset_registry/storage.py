"""Stage 0A — content-addressed physical storage for Master Dataset bytes.

Layout mirrors a git-object-store: <hash[:2]>/<hash>.<ext> under
settings.MASTER_DATASET_STORAGE_DIR. The original file extension is
preserved because Agent 1's loader (validate_and_load) branches on
filename extension to pick pd.read_csv vs pd.read_excel.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def master_file_path(base_dir: str, fingerprint: str, file_extension: str) -> Path:
    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
    return Path(base_dir) / fingerprint[:2] / f"{fingerprint}{ext}"


def write_master_file(base_dir: str, fingerprint: str, file_extension: str, content: bytes) -> str:
    """Atomic write: temp file in the same directory, then os.replace() —
    safe if two concurrent uploads of identical new content race each
    other (both write byte-identical content to the same final path;
    whichever replace() lands last wins, harmlessly, since the bytes are
    identical either way)."""
    path = master_file_path(base_dir, fingerprint, file_extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{fingerprint}.tmp"
    tmp_path.write_bytes(content)
    os.replace(tmp_path, path)
    return str(path)


def delete_master_file(storage_location: str) -> bool:
    """Best-effort delete — never raises. Called only after the DB row is
    already gone, so a leftover orphaned file is the safe failure mode,
    never a DB row pointing at nothing."""
    try:
        Path(storage_location).unlink(missing_ok=True)
        return True
    except OSError as e:
        logger.warning(f"Failed to delete master file at {storage_location}: {e}")
        return False
