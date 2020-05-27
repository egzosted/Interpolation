# Michał Piekarski 175456
import numpy as np
import copy
import numpy.linalg as la
import pandas as pd
from math_operations import Lagrange, LU, vector_norm, poly_val

profile = pd.read_csv("profile/Hel_yeah.csv", sep=",")
train_distance = []
train_altitude = []
index = 0
for i in profile["Distance"]:
    if index % 10 == 0:
        train_distance.append(i)
    index += 1

index = 0
for i in profile["Altitude"]:
    if index % 10 == 0:
        train_altitude.append(i)
    index += 1

# print(Lagrange(train_distance, train_altitude, len(train_distance)))

print(poly_val([3, 2, 8], 2))
