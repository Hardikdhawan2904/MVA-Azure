"""
In-memory DataFrame registry.
Stores loaded DataFrames keyed by dataset_id for downstream agent access.
"""

import pandas as pd
from typing import Optional

# Global in-memory store for DataFrames
_dataframes: dict[str, pd.DataFrame] = {}


def store_dataframe(dataset_id: str, df: pd.DataFrame) -> None:
    """
    Store a DataFrame in the in-memory registry.

    Args:
        dataset_id: Unique dataset identifier.
        df: The Pandas DataFrame to store.
    """
    _dataframes[dataset_id] = df


def get_dataframe(dataset_id: str) -> Optional[pd.DataFrame]:
    """
    Retrieve a DataFrame from the in-memory registry.

    Args:
        dataset_id: Unique dataset identifier.

    Returns:
        The stored DataFrame, or None if not found.
    """
    return _dataframes.get(dataset_id)
