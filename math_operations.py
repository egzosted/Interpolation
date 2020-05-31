import numpy as np
from math import sqrt
import copy

"""
    Class that represents Lagrange interpolation for certain train data
"""


class Lagrange:
    def __init__(self, X, Y, size):
        self.train_distance = X
        self.train_elevation = Y
        self.size = size

    """
        method that uses Lagrange interpolation to estimate value in list of arguments
        @param test_distance points that we want to calculate value in
        @return interpolated values
    """

    def interpolate(self, test_distance):
        interpolated_elevation = np.zeros(shape=[len(test_distance), 1])
        for i in range(len(test_distance)):
            for index, value in enumerate(self.train_distance):
                base = 1
                for j in range(self.size):
                    xj = float(self.train_distance[j])
                    if index != j:
                        base *= (test_distance[i] - xj) / (value - xj)
                interpolated_elevation[i] += self.train_elevation[index] * base
        return interpolated_elevation


"""
    Class that represents Spline interpolation for certain train data
"""


class Spline:
    def __init__(self, X, Y, size):
        self.train_distance = X
        self.train_elevation = Y
        self.size = size
        self.coefficients = np.zeros(shape=[4 * (size - 1), 4 * (size - 1)])
        self.free_variables = np.zeros(shape=[4 * (size - 1), 1])

        # size - 1 equations for Sj(xj) = f(xj)
        # size - 1 equations for Sj(xj + 1) = f(xj + 1)
        for i in range(size - 1):
            self.coefficients[i][4 * i] = 1
            self.coefficients[i][4 * i + 1] = 0
            self.coefficients[i][4 * i + 2] = 0
            self.coefficients[i][4 * i + 3] = 0
            h = self.train_distance[i + 1] - self.train_distance[i]
            self.coefficients[size - 1 + i][4 * i] = 1
            self.coefficients[size - 1 + i][4 * i + 1] = h
            self.coefficients[size - 1 + i][4 * i + 2] = h * h
            self.coefficients[size - 1 + i][4 * i + 3] = h * h * h
            self.free_variables[i] = self.train_elevation[i]
            self.free_variables[size - 1 + i] = self.train_elevation[i + 1]

        # # size - 2 equations for S'j-1(xj) = S'j(xj)
        # # size - 2 equations for S"j-1(xj) = S"j(xj)
        for i in range(1, size - 1):
            h = self.train_distance[i] - self.train_distance[i - 1]
            self.coefficients[2 * (size - 1) + i - 1][4 * (i - 1) + 1] = 1
            self.coefficients[2 * (size - 1) + i - 1][4 * (i - 1) + 2] = 2 * h
            self.coefficients[2 * (size - 1) + i - 1][4 * (i - 1) + 3] = 3 * h * h
            self.coefficients[2 * (size - 1) + i - 1][4 * (i - 1) + 5] = -1
            self.coefficients[3 * (size - 1) + i - 2][4 * (i - 1) + 2] = 2
            self.coefficients[3 * (size - 1) + i - 2][4 * (i - 1) + 3] = 6 * h
            self.coefficients[3 * (size - 1) + i - 2][4 * (i - 1) + 6] = -2

        # # 2 equations for corner values
        self.coefficients[4 * (size - 1) - 2][2] = 2
        self.coefficients[4 * (size - 1) - 1][4 * (size - 1) - 1] = 6 * h
        self.coefficients[4 * (size - 1) - 1][4 * (size - 1) - 2] = 2

        self.polynomial = LU(self.coefficients, self.free_variables, len(self.free_variables))

    def interpolate(self, test_distance):
        results = np.zeros(shape=[len(test_distance), 1])
        for i in range(len(test_distance)):
            for j in range(len(self.train_distance) - 1):
                if test_distance[i] >= self.train_distance[j] and test_distance[i] <= self.train_distance[j + 1]:
                    x0 = self.train_distance[j]
                    results[i] = poly_val(self.polynomial, test_distance[i] - x0, 4 * j)
                    break
        return results

        """
function that calculates value of polynomial from vector
@param polynomial vector of coefficients
@param arg, point that we want to calculate value in
@return value of polynomial in certain point
"""


def poly_val(polynomial, arg, j):
    value = 0
    index = 0
    for i in range(j, j + 4):
        value += polynomial[i] * (arg ** index)
        index += 1
    return value


"""
        function that calculates RMSD from cofficients of polynomial
        @param polynomial vector of coefficients
        @param test_distance list of arguments from function to interpolate
        @param test_altitude list of values from function to interpolate
        @return value of polynomial in certain point
"""


def RMSD(interpolated_elevation, test_elevation):
    rmsd = 0
    size = len(interpolated_elevation)
    for i in range(size):
        rmsd += (interpolated_elevation[i] - test_elevation[i]) ** 2
    rmsd = sqrt(rmsd / size)
    return rmsd


"""
           function that calculates system of linear equations using LU factorization
           @param A matrix of coefficients
           @param b vector of free variables
           @param size size of square matrix and vector
           @retur: solution vector
 """


def LU(A, b, size):
    U = copy.deepcopy(A)
    L = np.eye(size)
    P = np.eye(size)  # permutation matrix

    for j in range(size - 1):
        # finding pivot
        max = -1
        ind = 0
        for i in range(j, size):
            if abs(U[i][j]) > max:
                ind = i
                max = abs(U[i][j])

        # interchange rows
        for i in range(size):
            P[j][i], P[ind][i] = P[ind][i], P[j][i]

        for i in range(j):
            L[j][i], L[ind][i] = L[ind][i], L[j][i]

        for i in range(j, size):
            U[j][i], U[ind][i] = U[ind][i], U[j][i]

        for k in range(j + 1, size):
            L[k][j] = U[k][j] / U[j][j]
            for i in range(j, size):
                U[k][i] = U[k][i] - L[k][j] * U[j][i]

    y = forward_substitution(L, np.dot(P, b), size)
    x = backward_substitution(U, y, size)
    return x


"""
        function that calculates system of linear equations with forward substituion
        @param A matrix of coefficients
        @param b vector of free variables
        @param size size of square matrix and vector
        @return vector with solution
"""


def forward_substitution(A, b, size):
    x = np.zeros(shape=[size, 1])
    for i in range(size):
        sum = 0
        for j in range(i):
            sum += A[i][j] * x[j]
        x[i] = (b[i] - sum) / A[i][i]
    return x

    """
        function that calculates system of linear equations with forward substituion
        @param A matrix of coefficients
        @param b vector of free variables
        @param size size of square matrix and vector
        @return vector with solution
    """


def backward_substitution(A, b, size):
    x = np.zeros(shape=[size, 1])
    for i in range(size - 1, -1, -1):
        sum = 0
        for j in range(i + 1, size):
            sum += A[i][j] * x[j]
        x[i] = (b[i] - sum) / A[i][i]
    return x


"""
    function that calculates norm of vector
    @param vec vector that norm we want to calculate
    @return scalar, norm of vector
"""


def vector_norm(vec):
    norm = 0
    sum = 0
    for i in vec:
        sum += i * i
    norm = np.sqrt(sum)
    return norm
