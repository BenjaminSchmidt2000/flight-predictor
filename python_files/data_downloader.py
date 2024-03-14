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
