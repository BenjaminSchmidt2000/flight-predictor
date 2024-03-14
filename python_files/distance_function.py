import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of latitude and longitude coordinates.
    input:
    -param lat1: Latitude of the first point.
    -param lon1: Longitude of the first point.
    -param lat2: Latitude of the second point.
    -param lon2: Longitude of the second point.
    -return: Haversine distance in kilometers.
    """
    for coordinate in [lat1, lon1, lat2, lon2]:
        if type(coordinate) not in [int, float]:
            raise TypeError("All inputs must be numeric.")
    
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
