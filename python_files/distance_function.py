"""Module for calculating distances between two points on Earth."""
import math
from typing import Union

def haversine_distance(lat1: Union[int, float], lon1: Union[int, float],
                       lat2: Union[int, float], lon2: Union[int, float]) -> float:
    """
    Calculate the Haversine distance between two sets of latitude and longitude coordinates

    Parameters
    ---------------
    lat1 : float
        Latitude of the first point
    lon1 : float
        Longitude of the first point
    lat2 : float
        Latitude of the second point
    lon2 : float
        Longitude of the second point

    Returns
    ---------------
    float
        Haversine distance in kilometers
    """

    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
