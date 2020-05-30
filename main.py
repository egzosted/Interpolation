# Michał Piekarski 175456
import numpy as np
import copy
import pandas as pd
from math_operations import Lagrange, Spline, RMSD


profile = pd.read_csv("profile/plaska.csv", sep=",")
train_distance = []
train_elevation = []
test_distance = []
test_elevation = []
index = 0
for i in profile["Distance"]:
    if index % 5 < 5:
        train_distance.append(i)
    test_distance.append(i)
    index += 1

index = 0
for i in profile["Elevation"]:
    if index % 5 < 5:
        train_elevation.append(i)
    test_elevation.append(i)
    index += 1

spl = Spline(train_distance, train_elevation, len(train_distance))
res = spl.interpolate(test_distance)
print(res)
# pol = Lagrange(train_distance, train_elevation, len(train_distance))
# interpolated = pol.interpolate(test_distance)

# hit = 0
# for i in range(len(interpolated)):
#     if abs(interpolated[i] - test_elevation[i]) < 4.0:
#         print(i)
#         hit += 1
# print(hit * 100 / len(interpolated))
# print(RMSD(interpolated, train_elevation))
