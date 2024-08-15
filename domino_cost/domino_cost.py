import re
from datetime import timedelta
import pandas as pd
from pandas import (
    # DataFrame,
    Timestamp
)


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
