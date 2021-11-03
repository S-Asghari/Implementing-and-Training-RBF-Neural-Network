import math
import copy
import numpy as np

r = 1
mm = 2
c = 2

# -----------------------------------------------------------


def calculate_G(trainData, m, U, Vx, Vy):
    G = [[0 for x in range(m)] for y in range(len(trainData))]
    C = [[[0 for x in range(2)] for y in range(2)] for z in range(m)]

    for i in range(m):
        summ: float     # sum of membership values of all data to cluster i
        summ = 0
        for k in range(len(trainData)):
            summ += copy.deepcopy(pow(U[i][k], mm))

        # Calculating matrix C
        for k in range(len(trainData)):
            Xk_Vi = np.array([[trainData[k][0] - Vx[i]], [trainData[k][1] - Vy[i]]])
            C[i] += copy.deepcopy(pow(U[i][k], mm) * Xk_Vi.dot(Xk_Vi.transpose()))
        C[i] /= summ

        # Calculating matrix G
        for k in range(len(trainData)):
            Xk_Vi = np.array([[trainData[k][0] - Vx[i]], [trainData[k][1] - Vy[i]]])
            G[k][i] = math.exp(-1 * r * Xk_Vi.transpose().dot(np.linalg.inv(C[i]).dot(Xk_Vi)))

    G = np.array(G)
    return G

# -----------------------------------------------------------


def calculate_Y(trainData):
    Y = [[0 for x in range(c)] for y in range(len(trainData))]

    # Calculating matrix Y
    for i in range(len(trainData)):
        for j in range(1, c+1):
            if trainData[i][2] == j:
                Y[i][j-1] = 1

    Y = np.array(Y)

    return Y

# -----------------------------------------------------------


def calculate_W(m, G, Y):

    W = [[0 for x in range(c)] for y in range(m)]
    W = np.array(W)

    # Calculating matrix W
    W = np.linalg.inv(G.transpose().dot(G)).dot(G.transpose().dot(Y))

    return W

# -----------------------------------------------------------


def calculate_newY(G, W):

    GXW = np.array(G.dot(W))

    newY = [0 for x in range(len(G))]
    newY = np.array(newY)

    # calculating matrix newY
    for i in range(len(G)):
        result = np.where(GXW[i] == np.amax(GXW[i]))    # finding index of max value in each row of GXW
        newY[i] = result[0]

    return newY

# -----------------------------------------------------------


def accuracy(Y, newY):
    a = 0       # accuracy
    for i in range(len(newY)):
        if Y[i][newY[i]] == 1:
            a += 0
        else:
            a += 1
    a /= len(newY)
    a = 1 - a

    return a
