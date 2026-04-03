"""
cosmos.geo — Geospatial constellation (Python prototype)

Great-circle calculations, coordinate conversions, geodetic utilities.
All angles in degrees, all distances in metres unless noted.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "great_circle_distance_m",
    "bearing_deg",
    "destination_point",
    "midpoint",
    "area_of_polygon_m2",
    "degrees_to_radians",
    "radians_to_degrees",
    "dd_to_dms",
    "dms_to_dd",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
]

EARTH_RADIUS_M = 6_371_000.0    # mean Earth radius in metres
WGS84_A        = 6_378_137.0    # WGS-84 semi-major axis, m
WGS84_B        = 6_356_752.314  # WGS-84 semi-minor axis, m
WGS84_F        = 1.0 / 298.257223563  # flattening


def degrees_to_radians(deg: float) -> float:
    """Convert decimal degrees to radians."""
    return math.radians(float(deg))


def radians_to_degrees(rad: float) -> float:
    """Convert radians to decimal degrees."""
    return math.degrees(float(rad))


def great_circle_distance_m(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """
    Haversine great-circle distance on a sphere.
    Inputs in decimal degrees; output in metres.
    """
    lat1 = math.radians(float(lat1_deg))
    lon1 = math.radians(float(lon1_deg))
    lat2 = math.radians(float(lat2_deg))
    lon2 = math.radians(float(lon2_deg))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(float(radius_m) * c)


def bearing_deg(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
) -> float:
    """
    Initial bearing (forward azimuth) from point 1 to point 2, in degrees [0, 360).
    """
    lat1 = math.radians(float(lat1_deg))
    lat2 = math.radians(float(lat2_deg))
    dlon = math.radians(float(lon2_deg) - float(lon1_deg))
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return float(math.degrees(math.atan2(x, y)) % 360.0)


def destination_point(
    lat_deg: float, lon_deg: float,
    bearing_deg_val: float, distance_m: float,
    radius_m: float = EARTH_RADIUS_M,
) -> Tuple[float, float]:
    """
    Destination (lat, lon) in degrees given start, bearing, and distance.
    """
    lat  = math.radians(float(lat_deg))
    lon  = math.radians(float(lon_deg))
    bear = math.radians(float(bearing_deg_val))
    d_r  = float(distance_m) / float(radius_m)
    lat2 = math.asin(
        math.sin(lat) * math.cos(d_r)
        + math.cos(lat) * math.sin(d_r) * math.cos(bear)
    )
    lon2 = lon + math.atan2(
        math.sin(bear) * math.sin(d_r) * math.cos(lat),
        math.cos(d_r) - math.sin(lat) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def midpoint(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
) -> Tuple[float, float]:
    """
    Spherical midpoint between two points; returns (lat, lon) in degrees.
    """
    lat1 = math.radians(float(lat1_deg))
    lon1 = math.radians(float(lon1_deg))
    lat2 = math.radians(float(lat2_deg))
    lon2 = math.radians(float(lon2_deg))
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    lat_m = math.atan2(
        math.sin(lat1) + math.sin(lat2),
        math.sqrt((math.cos(lat1) + bx)**2 + by**2)
    )
    lon_m = lon1 + math.atan2(by, math.cos(lat1) + bx)
    return (math.degrees(lat_m), math.degrees(lon_m))


def area_of_polygon_m2(
    lat_lon_pairs: list,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """
    Spherical excess area of a polygon given [(lat,lon), ...] in degrees.
    Uses the spherical polygon formula (L'Huilier's theorem simplified).
    """
    if len(lat_lon_pairs) < 3:
        raise ValueError("polygon requires at least 3 vertices")
    n = len(lat_lon_pairs)
    total = 0.0
    for i in range(n):
        lat1, lon1 = lat_lon_pairs[i]
        lat2, lon2 = lat_lon_pairs[(i + 1) % n]
        total += math.radians(lon2 - lon1) * (
            2.0 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2))
        )
    return float(abs(total) * radius_m**2 / 2.0)


def dd_to_dms(dd: float) -> Tuple[int, int, float]:
    """Decimal degrees → (degrees, minutes, seconds)."""
    d = int(dd)
    m = int((dd - d) * 60.0)
    s = (dd - d - m / 60.0) * 3600.0
    return (d, m, float(s))


def dms_to_dd(degrees: int, minutes: int, seconds: float) -> float:
    """Degrees, minutes, seconds → decimal degrees."""
    return float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float = 0.0
) -> Tuple[float, float, float]:
    """
    WGS-84 geodetic (lat, lon, altitude) → ECEF (x, y, z) in metres.
    """
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    e2  = 1.0 - (WGS84_B / WGS84_A)**2
    n   = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat)**2)
    x   = (n + float(alt_m)) * math.cos(lat) * math.cos(lon)
    y   = (n + float(alt_m)) * math.cos(lat) * math.sin(lon)
    z   = (n * (1.0 - e2) + float(alt_m)) * math.sin(lat)
    return (x, y, z)


def ecef_to_geodetic(
    x_m: float, y_m: float, z_m: float
) -> Tuple[float, float, float]:
    """
    ECEF (x, y, z) in metres → WGS-84 geodetic (lat_deg, lon_deg, alt_m).
    Uses iterative Bowring method.
    """
    x, y, z = float(x_m), float(y_m), float(z_m)
    e2 = 1.0 - (WGS84_B / WGS84_A)**2
    p  = math.sqrt(x*x + y*y)
    lon = math.atan2(y, x)
    # Initial estimate
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(10):
        n   = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat)**2)
        lat = math.atan2(z + e2 * n * math.sin(lat), p)
    n   = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat)**2)
    alt = p / max(math.cos(lat), 1e-10) - n
    return (math.degrees(lat), math.degrees(lon), float(alt))
