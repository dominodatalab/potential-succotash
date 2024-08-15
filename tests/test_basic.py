import pytest

from domino_cost.domino_cost import get_domino_namespace


class TestCostDashboard:
    def test_basic(self):
        """
        This is a startup test
        :return:
        """
        result = "startup test"
        assert result

    def test_get_namespace(self):
        host_name = "http://thisfrontend.my-platform:80"
        host_ns = get_domino_namespace(host_name)
        assert host_ns == "my-platform"
