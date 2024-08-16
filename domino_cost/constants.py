from __future__ import annotations

from enum import StrEnum


class CostEnums(StrEnum):
    pass


class CostLabels(CostEnums):
    BILLING_TAG = "BILLING TAG"
    ORGANIZATION = "ORGANIZATION"
    PROJECT_NAME = "PROJECT NAME"
    USER = "USER"


class CostFieldsLabels(CostEnums):
    CPU_COST = "CPU COST"
    GPU_COST = "GPU COST"
    COMPUTE_COST = "COMPUTE COST"
    MEMORY_COST = "MEMORY COST"
    STORAGE_COST = "STORAGE COST"
    ALLOC_COST = "ALLOC COST"


class CostAggregatedLabels(CostEnums):
    TOTAL_COST = "TOTAL COST"
    CLOUD_COST = "CLOUD COST"
    COMPUTE_COST = "COMPUTE COST"
    STORAGE_COST = "STORAGE COST"


NO_TAG = "No tag"

window_to_param = {
    "30d": "Last 30 days",
    "14d": "Last 14 days",
    "7d": "Last 7 days"
}
