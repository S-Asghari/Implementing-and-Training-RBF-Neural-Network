import numpy as np
import csv
import matplotlib.pyplot as plt
import FCM
import training
import random

a = 0.3     # Percentage of test data to total data
# m = 4       # the number of clusters

# -----------------------------------------------------------

reader = csv.reader(open("D:/dataset/2clstrain1500.csv"))
data = [raw for raw in reader]
# data = np.delete(data, slice(1, len(data), 2), 0)   # Removing blank lines in data set(for csv file 1 and 2)
for i in range(len(data)):
    data[i] = [float(j) for j in data[i]]

# change labels from -1 and 1 to 1 and 2 (just for 2clstrain1200.csv)
for i in range(len(data)):
    if data[i][2] == -1:
        data[i][2] = 1
    elif data[i][2] == 1:
        data[i][2] = 2

# -----------------------------------------------------------


def plot(data, Y, newY):

    for i in range(len(data)):

        if data[i][2] == 1 and Y[i][newY[i]] == 1:
            plt.scatter(data[i][0], data[i][1], c='blue', marker='x', s=15)

        elif data[i][2] == 2 and Y[i][newY[i]] == 1:
            plt.scatter(data[i][0], data[i][1], c='green', marker='x', s=15)

        elif data[i][2] == 3 and Y[i][newY[i]] == 1:
            plt.scatter(data[i][0], data[i][1], c='yellow', marker='x', s=15)

        elif data[i][2] == 4 and Y[i][newY[i]] == 1:
            plt.scatter(data[i][0], data[i][1], c='grey', marker='x', s=15)

        elif data[i][2] == 5 and Y[i][newY[i]] == 1:
            plt.scatter(data[i][0], data[i][1], c='pink', marker='x', s=15)

        else:
            # print("m is greater than 5.")
            plt.scatter(data[i][0], data[i][1], c='red', marker='x', s=15)

# -----------------------------------------------------------

np.random.shuffle(data)
testData = data[0:int(len(data)*a)]
trainData = data[int(len(data)*a):len(data)]

# -----------------------------------------------------------


# finding optimum value for m
def find_best_m(m):

    # finding the centers of clusters
    Vx, Vy, U = FCM.do_iteration(trainData, m)

    # -------------------------------------------------------

    # training
    G = training.calculate_G(trainData, m, U, Vx, Vy)
    Y = training.calculate_Y(trainData)
    W = training.calculate_W(m, G, Y)
    newY = training.calculate_newY(G, W)
    a = training.accuracy(Y, newY)
    # print(a)

    # -------------------------------------------------------

    # testing
    testU = [[0 for x in range(len(testData))] for y in range(m)]  # labels of test data set
    testU: [float]
    testU = FCM.update_labels(testData, Vx, Vy, testU, m)

    testG = training.calculate_G(testData, m, testU, Vx, Vy)
    testY = training.calculate_Y(testData)
    test_newY = training.calculate_newY(testG, W)
    a2 = training.accuracy(testY, test_newY)
    # print(a2)

    return a, a2, Vx, Vy, testY, test_newY

best_val = 0
best_m = 0
for m in range(2, 10+1, 2):
    a1, a2, Vx, Vy, testY, test_newY = find_best_m(m)
    if best_val < a1+a2:
        best_val = a1+a2
        best_m = m

print("best value for m is: " + str(best_m))

# -----------------------------------------------------------

# plotting test data set
a1, a2, Vx, Vy, testY, test_newY = find_best_m(best_m)
print("accuracy of train data: " + str(a1))
print("accuracy of test data: " + str(a2))
plot(testData, testY, test_newY)
for i in range(best_m):
    plt.scatter(Vx[i], Vy[i], c='black', marker='x', s=15)
plt.show()

# -----------------------------------------------------------


# setting boundaries for clusters
def boundaries():
    randomData = [[0 for x in range(3)] for y in range(10000)]
    randomData_U = [[0 for x in range(len(randomData))] for y in range(best_m)]  # labels for random data set

    x_min, x_max, y_min, y_max = FCM.data_range(data)
    for i in range(len(randomData)):
        randomData[i][0] = x_min + (x_max - x_min) * random.uniform(0, 1)
        randomData[i][1] = y_min + (y_max - y_min) * random.uniform(0, 1)

    randomData_U = FCM.update_labels(randomData, Vx, Vy, randomData_U, best_m)

    for i in range(len(randomData)):
        max_val = 0
        for j in range(best_m):
            if max_val < randomData_U[j][i]:
                randomData[i][2] = j+1
                max_val = randomData_U[j][i]

    return randomData

# -----------------------------------------------------------


def plot_randomData(randomData):
    for i in range(len(randomData)):

        if randomData[i][2] == 1:
            plt.scatter(randomData[i][0], randomData[i][1], c='blue', marker='x', s=15)

        elif randomData[i][2] == 2:
            plt.scatter(randomData[i][0], randomData[i][1], c='green', marker='x', s=15)

        elif randomData[i][2] == 3:
            plt.scatter(randomData[i][0], randomData[i][1], c='yellow', marker='x', s=15)

        elif randomData[i][2] == 4:
            plt.scatter(randomData[i][0], randomData[i][1], c='grey', marker='x', s=15)

        elif randomData[i][2] == 5:
            plt.scatter(randomData[i][0], randomData[i][1], c='pink', marker='x', s=15)

        elif randomData[i][2] == 6:
            plt.scatter(randomData[i][0], randomData[i][1], c='purple', marker='x', s=15)

        elif randomData[i][2] == 7:
            plt.scatter(randomData[i][0], randomData[i][1], c='orange', marker='x', s=15)

        elif randomData[i][2] == 8:
            plt.scatter(randomData[i][0], randomData[i][1], c='brown', marker='x', s=15)

        elif randomData[i][2] == 9:
            plt.scatter(randomData[i][0], randomData[i][1], c='cyan', marker='x', s=15)

        elif randomData[i][2] == 10:
            plt.scatter(randomData[i][0], randomData[i][1], c='olive', marker='x', s=15)

# -----------------------------------------------------------

randomData = boundaries()
plot_randomData(randomData)
plt.show()
