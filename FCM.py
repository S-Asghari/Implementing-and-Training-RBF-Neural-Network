import random
import copy
import math

mm = 2      # power of Uik in the FCM algorithm
n = 100     # the number for FCM iteration

# -----------------------------------------------------------


# finding range of train data set
def data_range(trainData):
    x_min = float('Inf')
    y_min = float('Inf')
    x_max = 0
    y_max = 0

    for i in range(len(trainData)):
        if trainData[i][0] < x_min:
            x_min = trainData[i][0]

        if trainData[i][0] > x_max:
            x_max = trainData[i][0]

        if trainData[i][1] < y_min:
            y_min = trainData[i][1]

        if trainData[i][1] > y_max:
            y_max = trainData[i][1]

    return x_min, x_max, y_min, y_max

# -----------------------------------------------------------


# Initializing FCM centers
def initialize_centers(x_min, x_max, y_min, y_max, m):
    Vx = [None]*m   # x axis of FCM centers
    Vy = [None]*m   # y axis of FCM centers
    Vx: [float]
    Vy: [float]
    for i in range(m):
        Vx[i] = x_min + (x_max - x_min)*random.uniform(0, 1)
        Vy[i] = y_min + (y_max - y_min)*random.uniform(0, 1)

    return Vx, Vy

# -----------------------------------------------------------


# updating the labels
def update_labels(trainData, Vx, Vy, U, m):
    for k in range(len(trainData)):
        summ: float       # sum of all membership values of data k
        summ = 0
        for j in range(m):
            summ += copy.deepcopy(1 / pow(math.sqrt(pow(trainData[k][0] - Vx[j], 2) + pow(trainData[k][1] - Vy[j], 2)), (2 / (mm - 1))))
        for i in range(m):
            U[i][k] = copy.deepcopy(1 / pow(math.sqrt(pow(trainData[k][0] - Vx[i], 2) + pow(trainData[k][1] - Vy[i], 2)), (2 / (mm - 1))) / summ)

    return U


# updating the centers
def update_centers(trainData, Vx, Vy, U, m):
    for i in range(m):
        summ2: float      # sum of membership values of all data to cluster i
        summ2 = 0
        for k in range(len(trainData)):
            summ2 += copy.deepcopy(pow(U[i][k], mm))
        Vx[i] = 0
        Vy[i] = 0
        for k in range(len(trainData)):
            Vx[i] += copy.deepcopy(pow(U[i][k], mm) * trainData[k][0])
            Vy[i] += copy.deepcopy(pow(U[i][k], mm) * trainData[k][1])
        Vx[i] /= summ2
        Vy[i] /= summ2

    return Vx, Vy

# -----------------------------------------------------------


def do_iteration(trainData, m):
    x_min, x_max, y_min, y_max = data_range(trainData)
    Vx, Vy = initialize_centers(x_min, x_max, y_min, y_max, m)
    U = [[0 for x in range(len(trainData))] for y in range(m)]
    U: [float]
    for i in range(n):
        U = update_labels(trainData, Vx, Vy, U, m)
        Vx, Vy = update_centers(trainData, Vx, Vy, U, m)

    return Vx, Vy, U
