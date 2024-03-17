import warnings

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from urllib.request import urlretrieve
from zipfile import ZipFile
from pydantic import BaseModel, Field
import math
from .distance_function import haversine_distance
import contextily as ctx
from IPython.display import Markdown
import openai
from langchain_openai import OpenAI
from typing import Optional, Any


warnings.filterwarnings("ignore")


class DataDownloader(BaseModel):
    data_url: str
    file_name: str
    downloads_dir: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "downloads")
    )
    zip_dir: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "downloads", "zip_files")
    )
    airlines_df: pd.DataFrame = None
    airplanes_df: pd.DataFrame = None
    airports_df: pd.DataFrame = None
    routes_df: pd.DataFrame = None
    OPENAI_API_KEY: Optional[str] = None
    llm: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.setup_directories()
        self.initialize_downloader()
        self.load_datasets()
        self.validate_api_key()

    def setup_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.downloads_dir, self.zip_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def initialize_downloader(self) -> None:
        """Handle downloading, unzipping, and loading data."""
        if not os.path.isfile(os.path.join(self.downloads_dir, self.file_name)):
            self.download_data()
            self.unzip_data()
            self.validate_api_key()

    def download_data(self) -> None:
        """Implement data downloading logic."""
        # Placeholder for download logic
        print(f"Downloading data from {self.data_url}...")

    def unzip_data(self) -> None:
        """Implement data unzipping logic."""
        # Placeholder for unzip logic
        print("Unzipping data...")

    def load_datasets(self) -> None:
        """Load datasets into pandas DataFrames."""
        self.airlines_df = pd.read_csv(os.path.join(self.zip_dir, "airlines.csv")).drop(
            columns=["index"], axis=1
        )
        self.airplanes_df = pd.read_csv(
            os.path.join(self.zip_dir, "airplanes.csv")
        ).drop(columns=["index"], axis=1)
        self.airports_df = pd.read_csv(os.path.join(self.zip_dir, "airports.csv")).drop(
            columns=["index", "Type", "Source"], axis=1
        )
        self.routes_df = pd.read_csv(os.path.join(self.zip_dir, "routes.csv")).drop(
            columns=["index"], axis=1
        )

    def validate_api_key(self) -> None:
        """Validate the presence of an API key."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set.")
        else:
            self.llm = OpenAI(temperature=0.1)

    def download_data(self) -> None:
        file_path = os.path.join(self.downloads_dir, self.file_name)
        urlretrieve(self.data_url, file_path)
        print(f"Downloaded {self.file_name} to {self.downloads_dir}")

    def unzip_data(self) -> None:
        zip_file_path = os.path.join(self.downloads_dir, self.file_name)
        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.zip_dir)
        print(f"Unzipped {self.file_name} to {self.zip_dir}")

    def airport_distance(self, airport1: str = "", airport2: str = "") -> None:
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
        lat1: float = float(self.airports_df[self.airports_df["Name"] == airport1].iloc[0, 6])
        lat2: float = float(self.airports_df[self.airports_df["Name"] == airport2].iloc[0, 6])
        lon1: float = float(self.airports_df[self.airports_df["Name"] == airport1].iloc[0, 7])
        lon2: float = float(self.airports_df[self.airports_df["Name"] == airport2].iloc[0, 7])
        distance: float = haversine_distance(lat1, lon1, lat2, lon2)
        print(f"{distance} km")


    def plot_airports_map(self, country: str = "") -> None:
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

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        ax = world.plot(color="white", edgecolor="black", figsize=(10, 6))

        country_map = gpd.GeoDataFrame(
            country_airports,
            geometry=gpd.points_from_xy(
                country_airports.Longitude, country_airports.Latitude
            ),
        )
        country_map.plot(ax=ax, color="red", markersize=10)
        plt.title(f"Airports in {country}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def distance_analysis(self) -> None:
        """
        Plot the distribution of flight distances for all flights.
        """
        distances = []
        for index, row in self.routes_df.iterrows():
            source_airport = row["Source airport"]
            destination_airport = row["Destination airport"]
            source_info = self.airports_df[self.airports_df["IATA"] == source_airport]
            destination_info = self.airports_df[
                self.airports_df["IATA"] == destination_airport
            ]
            if not source_info.empty and not destination_info.empty:
                source_coords = (
                    float(source_info.iloc[0]["Latitude"]),
                    float(source_info.iloc[0]["Longitude"]),
                )
                destination_coords = (
                    float(destination_info.iloc[0]["Latitude"]),
                    float(destination_info.iloc[0]["Longitude"]),
                )
                distance = haversine_distance(*source_coords, *destination_coords)
                distances.append(distance)
        plt.hist(distances, bins=20, color="skyblue", edgecolor="black")
        plt.xlabel("Distance (km)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Flight Distances")
        plt.show()

    def plot_flights(self, airport, internal=False, fig=None, ax=None) -> None:
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

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        airport_point = Point(
            self.airports_df[self.airports_df["IATA"] == airport]["Longitude"].iloc[0],
            self.airports_df[self.airports_df["IATA"] == airport]["Latitude"].iloc[0],
        )

        if internal:
            airport_country = self.airports_df[self.airports_df["IATA"] == airport][
                "Country"
            ].iloc[0]
            internal_routes = self.routes_df[
                (self.routes_df["Source airport"] == airport)
                & (
                    self.routes_df["Destination airport"].isin(
                        self.airports_df[
                            self.airports_df["Country"] == airport_country
                        ]["IATA"]
                    )
                )
            ]
            internal_routes = internal_routes.merge(
                self.airports_df[["IATA", "Latitude", "Longitude", "Country"]],
                left_on="Destination airport",
                right_on="IATA",
                how="inner",
            )

            ax.set_title(f"Internal Flights from {airport}")
        else:
            all_routes = self.routes_df[self.routes_df["Source airport"] == airport]
            all_routes = all_routes.merge(
                self.airports_df[["IATA", "Latitude", "Longitude", "Country"]],
                left_on="Destination airport",
                right_on="IATA",
                how="inner",
            )

            ax.set_title(f"All Flights from {airport}")

        world.plot(ax=ax, color="lightgrey")
        ax.plot(airport_point.x, airport_point.y, "ro", markersize=5, label="Airport")

        for _, route in (
            internal_routes.iterrows() if internal else all_routes.iterrows()
        ):
            route_line = LineString(
                [
                    (airport_point.x, airport_point.y),
                    (route["Longitude"], route["Latitude"]),
                ]
            )
            ax.plot(*route_line.xy, "b-")

        ax.legend()
        return fig, ax

    def plot_top_airplane_models(self, countries=None, n=5) -> None:
        """
        Plot the N most used airplane models by number of routes.
        If countries are specified, plot only for those countries.
        If countries is None, plot for all dataset.
        :param countries: List of country names or None.
        :param n: Number of top airplane models to plot.
        """
        # Filter routes by countries if specified
        if isinstance(countries, str):
            countries = [countries]
        if isinstance(countries, list):
            filtered_routes = self.routes_df[
                (
                    self.routes_df["Source airport"].isin(
                        self.airports_df[self.airports_df["Country"].isin(countries)][
                            "IATA"
                        ]
                    )
                )
                & (
                    self.routes_df["Destination airport"].isin(
                        self.airports_df[self.airports_df["Country"].isin(countries)][
                            "IATA"
                        ]
                    )
                )
            ]
        else:
            filtered_routes = self.routes_df

        # Get the count of routes for each airplane model
        airplane_model_counts = filtered_routes["Equipment"].value_counts().head(n)

        # Plot the top N airplane models
        fig, ax = plt.subplots()
        airplane_model_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Airplane Model")
        ax.set_ylabel("Number of Routes")
        ax.set_title("Top {} Airplane Models by Number of Routes".format(n))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_country_flights(self, country, cutoff_distance: float, internal=False) -> None:
        """
        Plot flight routes within a specific country based on the given cutoff distance.

        Args:
            country (str): The country for which to plot the flight routes.
            cutoff_distance (float): The distance cutoff between short-haul and long-haul flights.
            internal (bool): Flag to indicate whether to consider only internal flights within the country.

        Returns:
            None
        """
        # Filter routes based on internal flag
        if internal:
            filtered_routes = self.routes_df[
                (self.routes_df["Source airport"].isin(self.airports_df[self.airports_df["Country"] == country]  ["IATA"]))
                & (self.routes_df["Destination airport"].isin(self.airports_df[self.airports_df["Country"] == country]["IATA"]))
            ]
        else:
            filtered_routes = self.routes_df[
                (self.routes_df["Source airport"].isin(self.airports_df[self.airports_df["Country"] == country]["IATA"]))
                | (self.routes_df["Destination airport"].isin(self.airports_df[self.airports_df["Country"] == country]["IATA"]))
            ]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, color="lightgrey")
        
        short_haul_params = {
                    'a': 0.00016,
                    'b': 1.454,
                    'c': 1531.722
                }

        long_haul_params = {
                    'a': 0.00034,
                    'b': 6.112,
                    'c': 3403.041
                }
                
        short_haul_count = 0
        short_haul_distance = 0
        total_flight_emissions_short = 0
        total_flight_emissions_long = 0
        processed_routes = set()
        
        # Define parameters for emission reduction by rail
        rail_emissions_ratio = 0.14
        
        for index, row in filtered_routes.iterrows():
            source_airport = row["Source airport"]
            destination_airport = row["Destination airport"]
            
            # Create a tuple representing the route pair
            route_pair = tuple(sorted([source_airport, destination_airport]))

            # Check if this route pair has already been processed
            if route_pair in processed_routes:
                continue

            processed_routes.add(route_pair)
    
            source_info = self.airports_df[self.airports_df["IATA"] == source_airport]
            destination_info = self.airports_df[self.airports_df["IATA"] == destination_airport]

            if not source_info.empty and not destination_info.empty:
                source_coords = (
                    float(source_info.iloc[0]["Latitude"]),
                    float(source_info.iloc[0]["Longitude"])
                )
                destination_coords = (
                    float(destination_info.iloc[0]["Latitude"]),
                    float(destination_info.iloc[0]["Longitude"])
                )

                distance = haversine_distance(*source_coords, *destination_coords)

                if distance > cutoff_distance:  # Check cutoff distance for defining short-haul flights
                    color = "orangered"  # Color for long-haul flights
                else:
                    color = "dodgerblue"  # Color for short-haul flights
                    

                if distance <= cutoff_distance:  # Short-haul flights
                    short_haul_count += 1
                    short_haul_distance += distance
                    params = short_haul_params
                    co2_emissions_short = params['a'] * distance ** 2 + params['b'] * distance + params['c']
                    total_flight_emissions_short += co2_emissions_short
                else:
                    params = long_haul_params
                    co2_emissions_long = params['a'] * distance ** 2 + params['b'] * distance + params['c']
                    total_flight_emissions_long += co2_emissions_long

                # Plot the flight route with the adjusted color
                ax.plot([source_coords[1], destination_coords[1]], [source_coords[0], destination_coords[0]], color=color, linewidth=2, alpha=0.5) 
    
            else:
                print(f"Route from {row['Source airport']} to {row['Destination airport']} skipped due to missing airport data.")

        # Plot airports
        airports_in_country = self.airports_df[self.airports_df["Country"] == country]
        ax.scatter(airports_in_country["Longitude"], airports_in_country["Latitude"], color="red", s=25, label="Airport")

        ax.set_title(f'{"Internal" if internal else "All"} Flights in {country}')
        ax.legend()
        
        # Calculate emission reduction for short-haul flights
        emission_reduction = total_flight_emissions_short * rail_emissions_ratio
        total_emissions_without_reduction = total_flight_emissions_short + total_flight_emissions_long
        total_flight_emissions = total_emissions_without_reduction- emission_reduction
        
        # Calculate the difference in emissions in percentage and absolute values
        emission_difference_percent = ((total_emissions_without_reduction - total_flight_emissions) / total_emissions_without_reduction) * 100
        emission_difference_abs = total_emissions_without_reduction - total_flight_emissions

        # Plot the calculations as annotations
        annotation_text = f"Short-Haul Flights: {short_haul_count}\nTotal Short-Haul Distance: {round(short_haul_distance, 2)} km\n"\
                  f"Emission Reduction with Rail Services: {emission_reduction:.2f} units\n"\
                  f"Emission Difference (Percentage): {emission_difference_percent:.2f}%\n"\
                  f"Emission Difference (Absolute): {emission_difference_abs:.2f} kg"

        plt.annotate(annotation_text, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.show()
    
    
    def airplanes(self) -> pd.Series:
        """
        Returns a pandas Series of all airplanes.

        Returns:
            pd.Series: A Series object containing the names of all airplanes.
        """
        return self.airplanes_df["Name"]

    def aircraft_info(self, aircraft_name: str) -> str:
        """
        Retrieves information about a specified aircraft using a language model and prints it in Markdown format.

        Parameters:
        aircraft_name (str): The name of the aircraft to retrieve information for.

        Raises:
        ValueError: If the specified aircraft name is not in the list of known aircrafts.

        Returns:
        str: A string containing a Markdown-formatted table of the aircraft specifications.
        """
        aircraft_name_normalized = aircraft_name.strip().lower()
        all_aircraft_names_normalized = self.airplanes().str.strip().str.lower()

        if aircraft_name_normalized not in all_aircraft_names_normalized.values:
            raise ValueError(
                f"{aircraft_name} is not a valid aircraft name. Please choose from the following: {', '.join(self.airplanes())}"
            )
        else:
            prompt = f"Generate a table in Markdown format with specifications and information for the aircraft named {aircraft_name}."
            result = self.llm(prompt)
            return Markdown(result)

    def airports(self) -> pd.Series:
        """
        Returns a pandas Series of all airports.

        Returns:
            pd.Series: A Series object containing the names of all airports.
        """
        return self.airports_df["Name"]

    def airport_info(self, airport_name: str) -> str:
        """
        Retrieves information about a specified aircraft using a language model and prints it in Markdown format.

        Parameters:
        aircraft_name (str): The name of the aircraft to retrieve information for.

        Raises:
        ValueError: If the specified aircraft name is not in the list of known aircrafts.

        Returns:
        str: A string containing a Markdown-formatted table of the aircraft specifications.
        """
        aircraft_name_normalized = airport_name.strip().lower()
        all_aircraft_names_normalized = self.airports().str.strip().str.lower()

        if aircraft_name_normalized not in all_aircraft_names_normalized.values:
            raise ValueError(
                f"{airport_name} is not a valid aircraft name. Please choose from the following: {', '.join(self.airports())}"
            )
        else:
            prompt = f"Generate a table in Markdown format with specifications and information for the airport named {airport_name}."
            result = self.llm(prompt)
            return Markdown(result)
