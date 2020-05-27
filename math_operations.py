import numpy as np
from math import sqrt
import copy

"""
        function that creates interpolation function
        @param X vector of points where we know correct value
        @param Y vector of values in point X
        @param size size of vectors X and Y
        @return interpolated polynomial
"""


def Lagrange(X, Y, size):
    # array of size - 1 basic functions (N funcs for N + 1 points)
    base_func = np.zeros(shape=[size, size])
    # array of denominators in lagrange polynomials
    denominators = np.zeros(shape=[size, 1])
    # array of coefficients in lagrange polynomials
    polynomial = np.zeros(shape=[size, 1])
    row = np.zeros(shape=[size, 1])

    for i in range(0, size):
        base_func[i][size - 1] = 1
        denominator = 1
        for j in range(0, size):
            if i != j:
                denominator *= X[i] - X[j]
                right = X[j] * -1
                for k in range(0, size):
                    base_func[i][k] = base_func[i][k] * right
                    if k != size - 1:
                        base_func[i][k] += base_func[i][k + 1]
        denominators[i] = denominator

    for i in range(0, size):
        for j in range(0, size):
            base_func[i][j] /= denominators[i]

    for i in range(0, size):
        row = base_func[i] * Y[i]
        for j in range(0, size):
            polynomial[j] += row[j]
    return polynomial


"""
        function that calculates value of polynomial from vector
        @param polynomial vector of coefficients
        @param arg, point that we want to calculate value in
        @return value of polynomial in certain point
"""


def poly_val(polynomial, arg):
    value = 0
    for i in range(len(polynomial)):
        value += polynomial[i] * (arg ** (len(polynomial) - i - 1))
    return value


"""
        function that calculates RMSD from cofficients of polynomial
        @param polynomial vector of coefficients
        @param test_distance list of arguments from function to interpolate
        @param test_altitude list of values from function to interpolate
        @return value of polynomial in certain point
"""


def RMSD(polynomial, test_distance, test_altitude):
    rmsd = 0
    for i in range(len(test_distance)):
        rmsd += (poly_val(polynomial, test_distance[i]) - test_altitude[i]) ** 2
        print(poly_val(polynomial, test_distance[i]))
        print(test_altitude[i])
    rmsd = sqrt(rmsd / len(test_distance))
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
    for k in range(size - 1):
        for j in range(k + 1, size):
            L[j][k] = U[j][k] / U[k][k]
            for i in range(k, size):
                U[j][i] = U[j][i] - L[j][k] * U[k][i]

    y = forward_substitution(L, b, size)
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
