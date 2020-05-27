# Michał Piekarski 175456
import numpy as np
import copy
import numpy.linalg as la
import pandas as pd
from math_operations import Lagrange, LU, vector_norm, poly_val, RMSD

profile = pd.read_csv("profile/small.csv", sep=",")
train_distance = []
train_altitude = []
index = 0
for i in profile["Distance"]:
    if index % 2 < 2:
        train_distance.append(i)
    index += 1

index = 0
for i in profile["Altitude"]:
    if index % 2 < 2:
        train_altitude.append(i)
    index += 1

print(train_distance)
pol = Lagrange(train_distance, train_altitude, len(train_distance))
# interp = []
# for i in profile["Distance"]:
#     interp.append(poly_val(pol, i))

# hit = 0
# index = 0
# for i in profile["Altitude"]:
#     if abs(i - interp[index]) < 1:
#         print(index)
#         hit += 1
#     index += 1
# print(hit * 100 / index)
print(RMSD(pol, train_distance, train_altitude))
