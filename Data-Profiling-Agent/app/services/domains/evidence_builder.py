"""Evidence builder for secondary-domain classification."""

from typing import Any

from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate


class DomainEvidence:
    """A single piece of evidence for domain classification."""

    def __init__(self, column: str, signal: str, weight: float = 1.0):
        self.column = column
        self.signal = signal
        self.weight = weight

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "signal": self.signal, "weight": self.weight}


class DomainScoreResult:
    """Score and evidence for a single secondary domain candidate."""

    def __init__(self, domain_name: str, score: float, evidence: list[DomainEvidence]):
        self.domain_name = domain_name
        self.score = score
        self.evidence = evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "score": self.score,
            "evidence": [e.to_dict() for e in self.evidence],
        }


class EvidenceBuilder:
    """Builds deterministic evidence for secondary-domain scoring."""

    def build_scores(
        self,
        secondary_domains_config: dict[str, Any],
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
    ) -> list[DomainScoreResult]:
        """
        Score each configured secondary domain based on deterministic evidence.

        Uses:
        - Column name keyword matches
        - Semantic type matches against configured roles
        - Representative value analysis
        """
        results: list[DomainScoreResult] = []

        for domain_name, domain_config in secondary_domains_config.items():
            keywords = [k.lower() for k in domain_config.get("keywords", [])]
            semantic_roles = [r.lower() for r in domain_config.get("semantic_roles", [])]

            evidence: list[DomainEvidence] = []
            total_score = 0.0

            for i, profile in enumerate(profiles):
                candidate = semantic_candidates[i] if i < len(semantic_candidates) else None
                col_name_lower = profile.normalized_key.lower()

                # Keyword match in column name
                for keyword in keywords:
                    if keyword in col_name_lower:
                        weight = 1.5
                        evidence.append(DomainEvidence(
                            column=profile.physical_name,
                            signal=f"name_keyword:{keyword}",
                            weight=weight,
                        ))
                        total_score += weight
                        break  # One match per column for name

                # Semantic role match
                if candidate and candidate.candidate_semantic_type:
                    sem_type_lower = candidate.candidate_semantic_type.lower()
                    for role in semantic_roles:
                        if role in sem_type_lower or sem_type_lower in role:
                            weight = 2.0
                            evidence.append(DomainEvidence(
                                column=profile.physical_name,
                                signal=f"semantic_role:{role}",
                                weight=weight,
                            ))
                            total_score += weight
                            break

                # Representative value keyword match
                for val in profile.representative_values[:5]:
                    val_lower = str(val).lower()
                    for keyword in keywords:
                        if keyword in val_lower:
                            weight = 0.5
                            evidence.append(DomainEvidence(
                                column=profile.physical_name,
                                signal=f"value_keyword:{keyword}",
                                weight=weight,
                            ))
                            total_score += weight
                            break
                    else:
                        continue
                    break  # One value match per column

            # Normalize score to [0, 1] using a sigmoid-like approach
            # Max possible score per column is ~4.0 (name + semantic + value)
            max_possible = len(profiles) * 4.0
            normalized_score = min(1.0, total_score / max(max_possible * 0.15, 1.0))

            results.append(DomainScoreResult(
                domain_name=domain_name,
                score=normalized_score,
                evidence=evidence,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results
