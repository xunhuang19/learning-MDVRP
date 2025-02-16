from .basics import ServicePoint, DictSet


class LoadingPoint(ServicePoint):

    def __init__(self, order_id, node_id=None, long=None, lat=None, tw_start=None, tw_end=None, duration=None):
        super(LoadingPoint, self).__init__(node_id, long, lat, tw_start, tw_end, duration)
        self.order_id = order_id


class UnloadingPoint(ServicePoint):

    def __init__(self, order_id, node_id=None, long=None, lat=None, tw_start=None, tw_end=None, duration=None):
        super(UnloadingPoint, self).__init__(node_id, long, lat, tw_start, tw_end, duration)
        self.order_id = order_id


class Order:
    """a base order contains id, volume, value and spatial info"""

    _total_order_had_automatically_named = 0

    def __init__(self, order_id=None, volume=None, value=None, loading_location: LoadingPoint = None,
                 unloading_location: UnloadingPoint = None):
        if order_id is None:
            Order._total_order_had_automatically_named += 1
            self.id = f"order_auto_named_{Order._total_order_had_automatically_named}"
        else:
            self.id = order_id
        self.volume = volume
        self.value = value
        self.loading_location = loading_location
        self.unloading_location = unloading_location

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.id}>"


class Task(DictSet):

    def __init__(self, data: dict[[], Order] = None, /, **kwargs):
        super(Task, self).__init__(data, **kwargs)

    def total_volume(self):
        return sum([order.volume] for order in self.values())

    def loading_location_ids(self):
        load_loc_ids = {}
        for order_id, order in self.items():
            load_loc = order.loading_location
            if load_loc is not None:
                load_loc_ids[order_id] = load_loc.node_id
        return load_loc_ids

    def loading_location_coords(self):
        load_loc_coords = {}
        for order_id, order in self.items():
            load_loc = order.loading_location
            if load_loc is not None:
                load_loc_coords[order_id] = load_loc.coordinates
        return load_loc_coords

    def unloading_location_ids(self):
        unload_loc_ids = {}
        for order_id, order in self.items():
            unload_loc = order.unloading_location
            if unload_loc is not None:
                unload_loc_ids[order_id] = unload_loc.node_id
        return unload_loc_ids

    def unloading_location_coords(self):
        unload_loc_coords = {}
        for order_id, order in self.items():
            unload_loc = order.unloading_location
            if unload_loc is not None:
                unload_loc_coords[order_id] = unload_loc.coordinates
        return unload_loc_coords
