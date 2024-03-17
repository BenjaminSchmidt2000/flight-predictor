# test_data_downloader.py

# Manual Unit test with correct distances out of this source ("https://www.luftlinie.org/9.652170181274414,%206.462259769439697/5-1-%E0%B8%AB%E0%B8%A1%E0%B8%B9%E0%B9%88-5-%E0%B8%95%E0%B8%B3%E0%B8%9A%E0%B8%A5%E0%B8%81%E0%B8%81%E0%B8%9B%E0%B8%A5%E0%B8%B2%E0%B8%8B%E0%B8%B4%E0%B8%A7-%E0%B8%AD%E0%B8%B3%E0%B9%80%E0%B8%A0%E0%B8%AD%E0%B8%A0%E0%B8%B9%E0%B8%9E%E0%B8%B2%E0%B8%99-%%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%AA%E0%B8%81%E0%B8%A5%E0%B8%99%E0%B8%84%E0%B8%A3-47180"):
"""
Unit test: comparing 3 long distances out of the model with the real distance taken out of the given source above:
first call function airport_distance
second put in two airport names 
third compare results with real distance calculated with source above
forth repeat three more times
finally evaluate how close the model's predictions are to the real distances
"""

import pytest
import numpy as np
import sys
sys.path.append("./python_files/")
from distance_function import haversine_distance

def test_one():
    with pytest.raises(TypeError):
        haversine_distance("Test")

def test_two():
    """Distance between Ronald Reagan Washington National Airport and Albury Airport"""
    assert haversine_distance(38.8521,-77.037697,-36.06779861450195,146.95799255371094) == pytest.approx(16157.03, abs=0.5)

def test_three():
    """Distance between Albury Airport and Belfast International Airport"""
    assert haversine_distance(-36.06779861450195,146.95799255371094,54.6575012207,-6.21582984924) == pytest.approx(17105.35, abs=0.5)

def test_four():
    """Distance between Belfast International Airport and Ronald Reagan Washington National Airport"""
    assert haversine_distance(54.6575012207,-6.21582984924,38.8521,-77.037697) == pytest.approx(5417.82, abs=0.5)
