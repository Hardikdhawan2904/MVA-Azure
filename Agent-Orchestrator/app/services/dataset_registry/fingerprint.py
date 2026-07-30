"""Stage 0A — content fingerprinting.

SHA-256 over the raw uploaded bytes — exact-match identity only. Per the
approved architecture doc: byte-identical content shares a Master Dataset;
any change at all (even one cell) is genuinely new content with its own
fingerprint. No fuzzy/near-duplicate similarity matching — deliberately
out of scope.
"""

import hashlib


def compute_fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
