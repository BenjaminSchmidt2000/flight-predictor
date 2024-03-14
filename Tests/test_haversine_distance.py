# test_data_downloader.py

import pytest
import numpy as np
import sys
sys.path.append("./python_files/")
from distance_function import haversine_distance

def test_haversine_distance_one():
    with pytest.raises(TypeError):
        haversine_distance("Test")

#def test_haversine_distance_two():


#def test_haversine_distance_three():