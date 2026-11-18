"""Geometry helpers.

Coordinates are lon/lat degrees. For distances we project each
trip to a local equirectangular plane in metres centred on the trip. Over a
city-sized area (< 50 km) the error versus geodesic distance is well below GPS
noise (< 0.1 %), and it lets us use fast vectorised Euclidean maths.
"""
import numpy as np

R_EARTH = 6371008.8


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.radians(1.0) * R_EARTH * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * R_EARTH

    def to_xy(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])

    def to_lonlat(self, xy):
        xy = np.asarray(xy)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0,
                                xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.
    Returns (distance, projection fraction in [0,1], unclamped fraction)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        u = ((px - ax) * dx + (py - ay) * dy) / L2
    u = np.where(L2 > 0, u, 0.0)
    uc = np.clip(u, 0.0, 1.0)
    qx, qy = ax + uc * dx, ay + uc * dy
    return np.hypot(px - qx, py - qy), uc, u


def bearing_xy(dx, dy):
    """Compass-style bearing in degrees [0,360) from planar displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
