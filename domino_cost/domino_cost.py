import re
from datetime import timedelta
import pandas as pd
from pandas import (
    DataFrame,
    Timestamp
)

from typing import Callable, List


def get_domino_namespace(api_host) -> str:
    pattern = re.compile("(https?://)((.*\.)*)(?P<ns>.*?):(\d*)\/?(.*)")
    match = pattern.match(api_host)
    return match.group("ns")


def get_today_timestamp() -> Timestamp:
    return pd.Timestamp("today", tz="UTC").normalize()


def get_time_delta(time_span) -> timedelta:
    if time_span == 'lastweek':
        days_to_use = 7
    else:
        days_to_use = int(time_span[:-1])
    return timedelta(days=days_to_use - 1)


def process_or_zero(func: Callable, posInt: int) -> int:
    if posInt > 0:
        return func
    else:
        return 0


def distribute_cost(df: DataFrame) -> DataFrame:
    """
    distributes __unallocated__ cost for cleaner representation.
    """
    fiels_list = ["CPU COST", "GPU COST", "COMPUTE COST", "MEMORY COST", "STORAGE COST", "ALLOC COST"]
    cost_unallocated = df[df["TYPE"].str.startswith("__")]
    cost_allocated = df[~df["TYPE"].str.startswith("__")]
    cost_allocated_total_sum = cost_allocated["ALLOC COST"].sum()

    for field in fiels_list:
        cost_allocated[field] = cost_allocated[field] + (
                    (cost_allocated["ALLOC COST"] / cost_allocated_total_sum) * cost_unallocated[field].sum())
    return cost_allocated


def distribute_cloud_cost(df: DataFrame, cost: float) -> DataFrame:
    """
    distribute unaccounted cloud cost accross allocated executions.
    """
    accounted_cost = df["ALLOC COST"].sum()
    cloud_cost_unaccounted = process_or_zero(cost - accounted_cost, cost)

    df['CLOUD COST'] = cloud_cost_unaccounted * (df['ALLOC COST'] / accounted_cost)
    df['TOTAL COST'] = df['ALLOC COST'] + df['CLOUD COST']

    return df


def clean_values(values_list: list) -> list:
    """
    remove "__unallocated__" from values'
    """
    return values_list[1:] if values_list[0].startswith("__") else values_list


def clean_df(df: DataFrame, col: str) -> DataFrame:
    """
    remove "__unallocated__" records from dataframe for preview.
    """
    return df[~df[col].str.startswith("__")]
