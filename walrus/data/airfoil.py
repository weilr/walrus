"""Airfoil entry points kept for configs written before the Geo-FNO generalization."""

from walrus.data.geofno import GeoFNODataModule, GeoFNONormalization

AirfoilNormalization = GeoFNONormalization


class AirfoilDataModule(GeoFNODataModule):
    def __init__(self, **kwargs):
        super().__init__(benchmark="airfoil", **kwargs)
