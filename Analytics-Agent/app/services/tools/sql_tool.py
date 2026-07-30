"""
app/services/sql_tool.py — DuckDB In-Process Query Engine

Executes SQL against the uploaded dataset via DuckDB.
The Analytics Agent uses this ONLY to retrieve required records.
Never performs calculations here — that is the Analytics Tool's job.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from app.config import DATASET_PATH

logger = logging.getLogger(__name__)


class SQLTool:
    """
    DuckDB-backed SQL executor.

    Rules:
    - Always returns a pandas DataFrame.
    - Never does calculations — only filtering, grouping, aggregation for retrieval.
    - Logs every query for auditability.

    One connection per instance, bound to `dataset_path` at construction —
    deliberately NOT a process-wide cached singleton. As a subprocess
    invoked fresh per upload this used to be safe to cache at module scope;
    as a long-lived FastAPI service handling many different uploads over
    its lifetime, a shared connection would silently keep answering every
    request from whichever dataset happened to be loaded first.
    """

    def __init__(self, dataset_path: str | Path | None = None, view_name: str = "insurance"):
        csv_path = str(dataset_path or DATASET_PATH)
        self.view_name = view_name
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"""
            CREATE VIEW {view_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', ALL_VARCHAR=FALSE, HEADER=TRUE)
        """)
        logger.info(f"DuckDB view '{view_name}' registered from: {csv_path}")

    # ── Core Executor ─────────────────────────────────────────────────────────

    def query(self, sql: str, parameters: list | None = None) -> pd.DataFrame:
        """Execute raw SQL and return a DataFrame. `parameters`, if given,
        binds to `?` placeholders in `sql` (DuckDB parameterized query) —
        used by the filter-building methods below so filter *values* never
        get string-interpolated directly into the SQL text."""
        logger.info(f"SQL >> {sql.strip()[:200]}")
        try:
            result = (
                self.conn.execute(sql, parameters) if parameters is not None
                else self.conn.execute(sql)
            ).fetchdf()
            logger.info(f"SQL returned {len(result)} rows, {len(result.columns)} columns")
            return result
        except Exception as e:
            logger.error(f"SQL error: {e}\nQuery: {sql}")
            raise

    # ── Convenience Builders ─────────────────────────────────────────────────

    def get_kpi_data(
        self,
        columns: list[str],
        filters: dict | None = None,
        group_by: list[str] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Generic KPI retrieval with optional filters and grouping.

        Args:
            columns   : List of column names to SELECT (can include SUM(x) etc.)
            filters   : Dict of {column: value} equality filters.
            group_by  : List of columns for GROUP BY.
            order_by  : ORDER BY clause string.
            limit     : Row limit.
        """
        select_clause = ", ".join(columns)
        sql = f"SELECT {select_clause} FROM {self.view_name}"

        # Column names come from code (KPI/hierarchy definitions), not user
        # input, so they stay as identifiers in the SQL text — only filter
        # *values* (which do trace back to query-derived dimension matches)
        # are bound as `?` parameters rather than string-interpolated.
        params: list = []
        if filters:
            conditions = []
            for col, val in filters.items():
                if val is None:
                    conditions.append(f"{col} IS NULL")
                elif isinstance(val, (list, tuple)):
                    conditions.append(f"{col} IN ({', '.join(['?'] * len(val))})")
                    params.extend(val)
                else:
                    conditions.append(f"{col} = ?")
                    params.append(val)
            sql += " WHERE " + " AND ".join(conditions)

        if group_by:
            sql += " GROUP BY " + ", ".join(group_by)

        if order_by:
            sql += f" ORDER BY {order_by}"

        if limit:
            sql += f" LIMIT {limit}"

        return self.query(sql, parameters=params or None)
