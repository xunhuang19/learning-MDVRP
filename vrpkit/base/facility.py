from .constant import INF
from .basics import ServicePoint, DictSet

from collections.abc import Iterable


class Vehicle:
    _total_vehicle_had_automatically_named = 0

    def __init__(self, vehicle_id=None, capacity=None, route_limit=None):
        if vehicle_id is None:
            Vehicle._total_vehicle_had_automatically_named += 1
            self.id = "vehicle_auto_named_" + str(Vehicle._total_vehicle_had_automatically_named)
        else:
            self.id = vehicle_id
        self.capacity = INF if capacity is None else capacity
        self.route_limit = INF if route_limit is None else route_limit


class DepotServicePoint(ServicePoint):

    def __init__(self, depot_id, node_id=None, long=None, lat=None, tw_start=None, tw_end=None, duration=None):
        super(DepotServicePoint, self).__init__(node_id, long, lat, tw_start, tw_end, duration)
        self.depot_id = depot_id


class Depot:
    _total_depot_had_automatically_named = 0

    def __init__(self, depot_id=None, location: DepotServicePoint = None):
        if depot_id is None:
            Depot._total_depot_had_automatically_named += 1
            self.id = "depot_auto_named_" + str(Depot._total_depot_had_automatically_named)
        else:
            self.id = depot_id
        self.location = location if location is not None else DepotServicePoint(depot_id)


class Fleet(DictSet):
    """a set of vehicles in a dict-like structure"""

    def __init__(self, data: dict[[], Vehicle] = None, /, **kwargs):
        super(Fleet, self).__init__(data, **kwargs)


class Depots(DictSet):
    """a set of Depots in a dict-like structure"""

    def __init__(self, data: dict[[], Depot] = None, /, **kwargs):
        super(Depots, self).__init__(data, **kwargs)

    def location_ids(self):
        return {depot_id: depot.location.node_id for depot_id, depot in self.items()}

    def location_coords(self):
        return {depot_id: depot.location.coordinates for depot_id, depot in self.items()}


class Asset:

    def __init__(self, depots: Depots = None, fleet: Fleet = None, halt: dict = None):
        self.depots = Depots() if depots is None else depots
        self.fleet = Fleet() if fleet is None else fleet
        self.halt = {} if halt is None else halt  # a mapping, {veh_id, depo_id}, indicating the halting scheme

    def __add__(self, other):
        return self.merge(other, replace=False)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        depots = self.depots - other.depots
        fleet = self.fleet - other.fleet
        halt = {v_i: self.halt[v_i] for v_i in fleet}
        return Asset(depots, fleet, halt)

    def copy(self):
        """self-defined shallow copy"""
        inst = Asset()
        inst.depots = Depots(self.depots)
        inst.fleet = Fleet(self.fleet)
        inst.halt = {}
        inst.halt.update(self.halt)
        return inst

    def merge(self, other: "Asset", replace=False):
        """Merge an asset of other company"""
        if not isinstance(other, Asset):
            raise ValueError(f"Can not merge {type(self)} with {type(other)}")

        if replace:
            self.depots += other.depots
            self.fleet += other.fleet
            self.halt.update(other.halt)
        else:
            depots = self.depots + other.depots
            fleet = self.fleet + other.fleet
            halt = {}
            halt.update(self.halt)
            halt.update(other.halt)
            return Asset(depots, fleet, halt)

    def subset(self, depot_ids=None, vehicle_ids=None):
        sub_asset = Asset()
        if vehicle_ids is not None:
            sub_asset.fleet = self.fleet.subset(vehicle_ids)
            sub_asset.halt = {v_i: self.halt[v_i] for v_i in sub_asset.fleet}
            if depot_ids is not None:
                sub_asset.depots = self.depots.subset(depot_ids)
                return sub_asset
            else:
                depot_ids = set([self.halt[v_i] for v_i in sub_asset.fleet])
                sub_asset.depots = self.depots.subset(depot_ids)
                return sub_asset
        else:
            if depot_ids is not None:
                sub_asset.depots = self.depots.subset(depot_ids)
                vehicle_ids = [v_i for v_i in self.fleet if self.halt[v_i] in sub_asset.depots.keys_in_list()]
                sub_asset.fleet = self.fleet.subset(vehicle_ids)
                sub_asset.halt = {v_i: self.halt[v_i] for v_i in vehicle_ids}
                return sub_asset
            else:
                return sub_asset

    def depot_vehicle_ids(self, depot_ids=None):
        all_depot_ids = self.depots.keys_in_list()
        if depot_ids is None:
            depot_ids = all_depot_ids
        elif not isinstance(depot_ids, Iterable) or isinstance(depot_ids, str):
            depot_ids = [depot_ids]
        depot2veh_ids = {depot_id: [] for depot_id in all_depot_ids}
        for veh_id, depot_id in self.halt.items():
            depot2veh_ids[depot_id].append(veh_id)
        return {depot_id: depot2veh_ids[depot_id] for depot_id in depot_ids}

    def depot_vehicles(self, depot_ids=None):
        depot2veh_ids = self.depot_vehicle_ids(depot_ids)
        depot2vehs = {}
        for depot_id, veh_ids in depot2veh_ids.items():
            depot2vehs[depot_id] = [self.fleet[i] for i in veh_ids]
        return depot2vehs

    def is_empty(self):
        return self.depots.is_empty() and self.fleet.is_empty()
