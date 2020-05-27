import numpy as np
import copy

"""
        function that creates interpolation function
        @param X vector of points where we know correct value
        @param Y vector of values in point X
        @param size size of vectors X and Y
        @return interpolated polynomial
"""


def Lagrange(X, Y, size):
    base_func = np.zeros(shape=[size, size])
    denominators = np.zeros(shape=[size, 1])
    polynomial = np.zeros(shape=[size, 1])
    row = np.zeros(shape=[size, 1])
    for i in range(0, size):
        denominator = 1
        for j in range(0, size):
            if i != j:
                denominator *= X[i] - X[j]
        denominators[i] = denominator

    for i in range(0, size):
        base_func[i][size - 1] = 1
        for j in range(0, size):
            if i != j:
                right = X[j] * -1
                for k in range(0, size):
                    base_func[i][k] = base_func[i][k] * right
                    if k != size - 1:
                        base_func[i][k] += base_func[i][k + 1]
    for i in range(0, size):
        for j in range(0, size):
            base_func[i][j] /= denominators[i]

    for i in range(0, size):
        row = base_func[i] * Y[i]
        for j in range(0, size):
            polynomial[j] += row[j]
    return polynomial


def poly_val(polynomial, arg):
    value = 0
    for i in range(len(polynomial)):
        value += polynomial[i] * (arg ** (len(polynomial) - i - 1))
    return value


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
