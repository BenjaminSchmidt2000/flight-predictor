import os
import pandas as pd
from urllib.request import urlretrieve
from zipfile import ZipFile
import math
from .distance_function import haversine_distance

class DataDownloader:
    def __init__(self, data_url, file_name):
        self.data_url = data_url
        self.file_name = file_name
        self.downloads_dir = os.path.join(os.getcwd(), 'downloads')
        self.zip_dir = os.path.join(self.downloads_dir, 'zip_files')

        # Check if directories exist, create if not
        for directory in [self.downloads_dir, self.zip_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Check if file already exists
        if not os.path.isfile(os.path.join(self.downloads_dir, self.file_name)):
            self.download_data()

        # Unzip the downloaded file
        self.unzip_data()

        # Read datasets into corresponding pandas dataframes
        self.airlines_df = pd.read_csv(os.path.join(self.zip_dir, 'airlines.csv')).drop(columns=["index"], axis=1)
        self.airplanes_df = pd.read_csv(os.path.join(self.zip_dir, 'airplanes.csv')).drop(columns=["index"], axis=1)
        self.airports_df = pd.read_csv(os.path.join(self.zip_dir, 'airports.csv')).drop(
            columns=["index", "Type", "Source"], axis=1)
        self.routes_df = pd.read_csv(os.path.join(self.zip_dir, 'routes.csv')).drop(columns=["index"], axis=1)

    def download_data(self):
        file_path = os.path.join(self.downloads_dir, self.file_name)
        urlretrieve(self.data_url, file_path)
        print(f"Downloaded {self.file_name} to {self.downloads_dir}")

    def unzip_data(self):
        zip_file_path = os.path.join(self.downloads_dir, self.file_name)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.zip_dir)
        print(f"Unzipped {self.file_name} to {self.zip_dir}")

    def airport_distance(self ,airport1="", airport2=""):
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
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        print(distance, "km")
         def plot_airports_map(self, country):
        """
        Plot a map with the locations of airports in the specified country.

        Parameters:
        - country (str): Name of the country.

        Returns:
        - None
        """
        country_airports = self.airports_df[self.airports_df["Country"] == country]
        if country_airports.empty:
            print("Error: Country does not exist or has no airports.")
            return
        
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = world.plot(color='white', edgecolor='black', figsize=(10, 6))

        country_map = gpd.GeoDataFrame(country_airports,
                                       geometry=gpd.points_from_xy(country_airports.Longitude,
                                                                    country_airports.Latitude))
        country_map.plot(ax=ax, color='red', markersize=10)
        plt.title(f'Airports in {country}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    
    def distance_analysis(self):
        """
        Plot the distribution of flight distances for all flights.
        """
        distances = []
        for index, row in self.routes_df.iterrows():
            source_airport = row['Source airport']
            destination_airport = row['Destination airport']
            source_info = self.airports_df[self.airports_df['IATA'] == source_airport]
            destination_info = self.airports_df[self.airports_df['IATA'] == destination_airport]
            if not source_info.empty and not destination_info.empty:
                source_coords = (source_info.iloc[0]['Latitude'], source_info.iloc[0]['Longitude'])
                destination_coords = (destination_info.iloc[0]['Latitude'], destination_info.iloc[0]['Longitude'])
                distance = self.haversine_distance(*source_coords, *destination_coords)
                distances.append(distance)
        plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Flight Distances')
        plt.show()

    def plot_flights(self, airport, internal=False, fig=None, ax=None):
        """
        Plot flights leaving the specified airport.

        Parameters:
        - airport (str): IATA code of the airport.
        - internal (bool): If True, plot only flights within the same country. Default is False.
        - fig (plt.figure, optional): Existing figure object to use for the plot.
        - ax (plt.axes, optional): Existing axes object to use for the plot.

        Returns:
        - plt.figure: Figure object containing the plot.
        - plt.axes: Axes object containing the plot.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        airport_point = Point(self.airports_df[self.airports_df['IATA'] == airport]['Longitude'].iloc[0], 
                              self.airports_df[self.airports_df['IATA'] == airport]['Latitude'].iloc[0])

        if internal:
            airport_country = self.airports_df[self.airports_df['IATA'] == airport]['Country'].iloc[0]
            internal_routes = self.routes_df[(self.routes_df['Source airport'] == airport) & 
                                             (self.routes_df['Destination airport'].isin(
                                                 self.airports_df[self.airports_df['Country'] == airport_country]['IATA']
                                             ))]
            internal_routes = internal_routes.merge(self.airports_df[['IATA', 'Latitude', 'Longitude', 'Country']], 
                                                    left_on='Destination airport', right_on='IATA', how='inner')

            ax.set_title(f'Internal Flights from {airport}')
        else:
            all_routes = self.routes_df[self.routes_df['Source airport'] == airport]
            all_routes = all_routes.merge(self.airports_df[['IATA', 'Latitude', 'Longitude', 'Country']], 
                                          left_on='Destination airport', right_on='IATA', how='inner')

            ax.set_title(f'All Flights from {airport}')

        world.plot(ax=ax, color='lightgrey')
        ax.plot(airport_point.x, airport_point.y, 'ro', markersize=5, label='Airport')

        for _, route in internal_routes.iterrows() if internal else all_routes.iterrows():
            route_line = LineString([(airport_point.x, airport_point.y), (route['Longitude'], route['Latitude'])])
            ax.plot(*route_line.xy, 'b-')

        ax.legend()
        return fig, ax
    
    def plot_top_airplane_models(self, countries=None, n=5):
        """
        Plot the N most used airplane models by number of routes.
        If countries are specified, plot only for those countries.
        If countries is None, plot for all dataset.
        :param countries: List of country names or None.
        :param n: Number of top airplane models to plot.
        """
        # Filter routes by countries if specified
        if countries:
            filtered_routes = self.routes_df[
                (self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'].isin(countries)]['IATA'])) &
                (self.routes_df['Destination airport'].isin(self.airports_df[self.airports_df['Country'].isin(countries)]['IATA']))
            ]
        else:
            filtered_routes = self.routes_df

        # Get the count of routes for each airplane model
        airplane_model_counts = filtered_routes['Equipment'].value_counts().head(n)

        # Plot the top N airplane models
        fig, ax = plt.subplots()
        airplane_model_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Airplane Model')
        ax.set_ylabel('Number of Routes')
        ax.set_title('Top {} Airplane Models by Number of Routes'.format(n))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_country_flights(self, country, internal=False):
        """
        Plot flights leaving or arriving in the specified country.
        :param country: Name of the country.
        :param internal: If True, plot only internal flights; otherwise, plot all flights.
        """
        if internal:
            filtered_routes = self.routes_df[
                (self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA'])) &
                (self.routes_df['Destination airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA']))
            ]
        else:
            filtered_routes = self.routes_df[
                (self.routes_df['Source airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA'])) |
                (self.routes_df['Destination airport'].isin(self.airports_df[self.airports_df['Country'] == country]['IATA']))
            ]

        # Get airport coordinates
        airport_coords = self.airports_df[self.airports_df['Country'] == country][['Longitude', 'Latitude']].values

        # Plot the flights on a map
        fig, ax = plt.subplots(figsize=(10, 8))
        for index, row in filtered_routes.iterrows():
            source = self.airports_df[self.airports_df['IATA'] == row['Source airport']].iloc[0]
            dest = self.airports_df[self.airports_df['IATA'] == row['Destination airport']].iloc[0]
            ax.plot([source['Longitude'], dest['Longitude']], [source['Latitude'], dest['Latitude']], color='blue')
        ax.scatter(airport_coords[:, 0], airport_coords[:, 1], color='red', label='Airports')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Flights in {}'.format(country))
        ax.legend()
        plt.show()
