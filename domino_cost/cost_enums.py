from enum import Enum


class StrEnum(str, Enum):  # Importing from newer enum version.
    """
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()


class CostEnums(StrEnum):
    @classmethod
    def to_values_list(cls):
        return list(map(lambda c: c.value, cls))


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

    def get_legend_labels(self):
        return list(map(lambda c: c.value, [self.CPU_COST, self.GPU_COST, self.STORAGE_COST]))


class CostAggregatedLabels(CostEnums):
    TOTAL_COST = "TOTAL COST"
    CLOUD_COST = "CLOUD COST"
    COMPUTE_COST = "COMPUTE COST"
    STORAGE_COST = "STORAGE COST"


class CostGraphFields(CostEnums):
    START_DATE = "START DATE"
    END = "END"
    TYPE = "TYPE"
    FORMATTED_START = "FORMATTED START"
    START = "START"
