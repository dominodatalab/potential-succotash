from datetime import timedelta
from enum import StrEnum

import pytest

from domino_cost.constants import CostEnums
from domino_cost.domino_cost import *


class TestCostDashboard:
    def test_get_namespace(self):
        host_name = "http://thisfrontend.my-platform:80"
        host_ns = get_domino_namespace(host_name)
        assert host_ns == "my-platform"

    def test_get_time_delta(self):
        input = "lastweek"
        expected = timedelta(days=6)
        assert get_time_delta(input) == expected
        assert get_time_delta("7d") == expected

    def test_process_or_zero(self):
        def func(x: int) -> Any:
            return x + 15

        assert process_or_zero(func(15), 15) == 30
        assert process_or_zero(func(0), 15) == 15
        assert process_or_zero(func(15), 0) == 0

    def test_clean_values(self):
        input_vals = ["__unallocated__", "one", "two", "three"]
        output_vals = ["one", "two", "three"]
        assert clean_values(input_vals) == output_vals


class TestConstants:
    def test_get_enums_list(self):
        class TestEnum(CostEnums):
            One = "one"
            Two = "two"

        assert TestEnum.to_values_list() == ["one", "two"]
