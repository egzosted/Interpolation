# Michał Piekarski 175456
import pandas as pd
from math_operations import Lagrange, Spline, RMSD
import matplotlib.pyplot as plt


profile = pd.read_csv("profile/kilka_wzniesien.csv", sep=",")
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
pol = Lagrange(train_distance, train_elevation, len(train_distance))
res2 = pol.interpolate(test_distance)
print(RMSD(res, test_elevation))


plt.title("Porownanie zaimplementowanych metod")
plt.xlabel("Dystans [m]")
plt.ylabel("Wysokosc [m]")
plt.semilogy(test_distance, test_elevation, 'r', label="Wysokość rzeczywista")
plt.semilogy(test_distance, res, 'g', label="Wysokość aproksymowana splajnami")
plt.semilogy(test_distance, res2, 'b', label="Wysokość aproksymowana Lagrangem")
plt.legend()
plt.show()
