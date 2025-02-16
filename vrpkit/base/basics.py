from .constant import INF

from collections import UserDict
from collections.abc import Iterable


class Point:
    """a physical point/node on a network"""

    total_node_had_automatically_named = 0

    def __init__(self, node_id=None, long=None, lat=None):
        if node_id is None:
            Point.total_node_had_automatically_named += 1
            self.node_id = "node_auto_named_" + str(Point.total_node_had_automatically_named)
        else:
            self.node_id = node_id
        self.long = long
        self.lat = lat

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.node_id}>"

    @property
    def coordinates(self):
        return self.long, self.lat


class TimeWindow:
    """time window for a pickup or delivery service"""

    def __init__(self, start=None, end=None):
        self.start = start if start is not None else 0
        self.end = end if end is not None else INF

    @property
    def time_window_in_tuple(self):
        return self.start, self.end


class ServicePoint(Point):
    def __init__(self, node_id=None, long=None, lat=None, tw_start=None, tw_end=None, duration=None):
        super(ServicePoint, self).__init__(node_id, long, lat)
        self.time_window = TimeWindow(tw_start, tw_end)
        self.service_duration = duration if duration is not None else 0


class DictSet(UserDict):
    "Dict-like data Structure supporting set operations"

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.__class__({**self.data, **other.data})
        elif isinstance(other, type(self.data)):
            return self.__class__({**self.data, **other})
        else:
            raise ValueError(f"adding between {type(self)} and {type(other)} is not supported")

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self.__class__({k: self.data[k] for k in self.data if k not in other.data})
        elif isinstance(other, type(self.data)):
            return self.__class__({k: self.data[k] for k in self.data if k not in other})
        else:
            raise ValueError(f"subtracting between {type(self)} and {type(other)} is not supported")

    def merge(self, other, replace=False):

        if replace:
            if isinstance(other, type(self)):
                self.data.update(other.data)
            if isinstance(other, type(self.data)):
                self.data.update(other)
            else:
                raise ValueError(f"can not merge {type(self)} with {type(other)}")
        else:
            return self.__add__(other)

    def copy(self):
        return self.__class__(self.data)

    def intersection(self, other):
        if isinstance(other, type(self)):
            return self.__class__({k: self.data[k] for k in self.data if k in other.data})
        elif isinstance(other, type(self.data)):
            return self.__class__({k: self.data[k] for k in self.data if k in other})
        else:
            raise ValueError(f"intersection between {type(self)} and {type(other)} is not supported")

    def union(self, other):
        return self.__add__(other)

    def subset(self, keys):
        if isinstance(keys, Iterable) and not isinstance(keys, str):
            return self.__class__({k: self.data[k] for k in keys})
        else:
            return self.__class__({keys: self.data[keys]})

    def keys_in_list(self):
        return list(self.keys())

    def values_in_list(self):
        return list(self.values())

    def is_empty(self):
        return len(self.data) == 0

    def get_attributes(self, att_name: str, keys=None):
        if keys is None:
            return {k: getattr(self[k], att_name) for k in self}
        else:
            if isinstance(keys, Iterable):
                return {k: getattr(self[k], att_name) for k in keys}
            else:
                return {keys: getattr(self[keys], att_name)}
