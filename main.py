# Michał Piekarski 175456
import pandas as pd
from math_operations import Lagrange, Spline, RMSD, LU
import numpy as np


profile = pd.read_csv("profile/plaska.csv", sep=",")
train_distance = []
train_elevation = []
test_distance = []
test_elevation = []
index = 0
for i in profile["Distance"]:
    if index % 5 == 0:
        train_distance.append(i)
    test_distance.append(i)
    index += 1

index = 0
for i in profile["Elevation"]:
    if index % 5 == 0:
        train_elevation.append(i)
    test_elevation.append(i)
    index += 1

spl = Spline(train_distance, train_elevation, len(train_distance))
res = spl.interpolate(test_distance)
print(RMSD(res, test_elevation))
# pol = Lagrange(train_distance, train_elevation, len(train_distance))
# interpolated = pol.interpolate(test_distance)

# hit = 0
# for i in range(len(res)):
#     if abs(res[i] - test_elevation[i]) < 0.3:
#         print(i)
#         hit += 1
# print(hit * 100 / len(res))
# print(RMSD(interpolated, train_elevation))
