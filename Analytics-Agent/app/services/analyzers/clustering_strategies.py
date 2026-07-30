"""app/services/analyzers/clustering_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): DBSCAN/Agglomerative clustering strategies
registered in config/model_registry.yml alongside K-Means (RiskSegmenter).

Both share `.cluster(df, feature_cols, label_cols=None)`, returning the
same shape RiskSegmenter.segment() already produces (cluster_profiles /
segment_distribution / records_with_segments) so ClusteringAnalyzer /
SegmentationAnalyzer don't need an algorithm-specific branch for these.
"""

from __future__ import annotations

import pandas as pd


def _feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Index, list[str]]:
    cols = [c for c in feature_cols if c in df.columns]
    feat = df[cols].apply(pd.to_numeric, errors="coerce")
    feat = feat.fillna(feat.median())
    valid_idx = feat.index
    return feat, valid_idx, cols


class DBSCANStrategy:
    """Density-based clustering — unlike K-Means, doesn't require choosing
    a cluster count upfront and naturally labels sparse regions as noise
    (cluster == -1) rather than forcing them into the nearest centroid."""

    def cluster(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, eps: float = 0.5, min_samples: int = 5) -> dict:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        feat, valid_idx, cols = _feature_frame(df, feature_cols)
        if not cols:
            return {"error": "No feature columns found in DataFrame for DBSCAN clustering."}
        if len(feat) < 10:
            return {"error": f"Insufficient rows for DBSCAN clustering: {len(feat)} (need >= 10)"}

        X = StandardScaler().fit_transform(feat)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        result_df = df.loc[valid_idx].copy()
        result_df["cluster"] = labels

        noise_points = int((labels == -1).sum())
        distribution = result_df["cluster"].value_counts().to_dict()

        profiles = []
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            profile = {"cluster": int(cluster_id), **{c: round(float(feat.loc[valid_idx][mask][c].mean()), 2) for c in cols}}
            profiles.append(profile)

        output_cols = list(label_cols or []) + ["cluster"]
        output_records = result_df[[c for c in output_cols if c in result_df.columns]].to_dict(orient="records")

        return {
            "evidence": {
                "model": "DBSCAN",
                "n_clusters": len([c for c in distribution if c != -1]),
                "noise_points": noise_points,
                "feature_columns": cols,
                "cluster_profiles": profiles,
                "segment_distribution": {int(k): int(v) for k, v in distribution.items()},
                "records_with_segments": output_records,
            },
            "metrics": {"eps": eps, "min_samples": min_samples},
        }


class AgglomerativeClusteringStrategy:
    """Hierarchical (bottom-up) clustering — reports the merge-distance
    sequence as lightweight "dendrogram" evidence rather than a rendered
    plot; ExplanationTool narrates the numbers, it doesn't draw pictures."""

    def cluster(self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str] | None = None, n_clusters: int = 3) -> dict:
        from scipy.cluster.hierarchy import linkage
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        feat, valid_idx, cols = _feature_frame(df, feature_cols)
        if not cols:
            return {"error": "No feature columns found in DataFrame for Agglomerative clustering."}
        if len(feat) < n_clusters * 2:
            return {"error": f"Too few records ({len(feat)}) for {n_clusters} clusters"}

        X = StandardScaler().fit_transform(feat)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

        result_df = df.loc[valid_idx].copy()
        result_df["cluster"] = labels
        distribution = result_df["cluster"].value_counts().to_dict()

        profiles = []
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            profile = {"cluster": int(cluster_id), **{c: round(float(feat.loc[valid_idx][mask][c].mean()), 2) for c in cols}}
            profiles.append(profile)

        # Ward linkage merge distances — the numeric backbone of a
        # dendrogram, capped to the last few merges (most informative).
        Z = linkage(X, method="ward")
        dendrogram_merges = [round(float(d), 4) for d in Z[-min(10, len(Z)):, 2]]

        output_cols = list(label_cols or []) + ["cluster"]
        output_records = result_df[[c for c in output_cols if c in result_df.columns]].to_dict(orient="records")

        return {
            "evidence": {
                "model": "Hierarchical Clustering",
                "n_clusters": n_clusters,
                "feature_columns": cols,
                "cluster_profiles": profiles,
                "segment_distribution": {int(k): int(v) for k, v in distribution.items()},
                "records_with_segments": output_records,
                "dendrogram_merge_distances": dendrogram_merges,
            },
        }
