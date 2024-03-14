def haversine_distance(self, lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of latitude and longitude coordinates.
    input:
    -param lat1: Latitude of the first point.
    -param lon1: Longitude of the first point.
    -param lat2: Latitude of the second point.
    -param lon2: Longitude of the second point.
    -return: Haversine distance in kilometers.
    """
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def airport_distance(self, airport1="", airport2=""):
    """
    Extract the latitude and longitude for the two chosen airports to then call the haversine_distance method
    which then calculates the distance of the chosen airports.
    input:
    -param airport1
    -param airport2
    output:
    -param lat1: Latitude of the first point.
    -param lon1: Longitude of the first point.
    -param lat2: Latitude of the second point.
    -param lon2: Longitude of the second point.
    call haversine_distance(lat1, lon1, lat2, lon2) with output parameter
    """
    lat1 = float(self.airports_df[self.airports_df["Name"] == airport1].iloc[:, 6])
    lat2 = float(self.airports_df[self.airports_df["Name"] == airport2].iloc[:, 6])
    lon1 = float(self.airports_df[self.airports_df["Name"] == airport1].iloc[:, 7])
    lon2 = float(self.airports_df[self.airports_df["Name"] == airport2].iloc[:, 7])
    distance = self.haversine_distance(lat1, lon1, lat2, lon2)
    print(distance, "km")