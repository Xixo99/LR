import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    x_train = np.load('./data/LR/train_data.npy')
    y_train = np.load('./data/LR/train_target.npy')
    x_test = np.load('./data/LR/test_data.npy')
    y_test = np.load('./data/LR/test_target.npy')

    return x_train, y_train, x_test, y_test

def sigmod(Z):
    return 1.0 / (1+ np.exp(-Z))

def thanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def gradAscent(dataSet,label):
    dataMat = np.mat(dataSet)   #  (m,n)
    m, n = np.shape(dataMat)
    lableMat = np.mat(label).transpose()  ## (m,1)

    alpha = 0.001
    maxCycle = 1000000

    weights = np.ones((n,1))
    for i in range(maxCycle):
        h = sigmod(dataMat*weights)  #   ( m,1 )
        #h = thanh(dataMat*weights)
        error = (lableMat-h)     # (m,1)
        weights = weights + alpha * dataMat.transpose() * error

    return np.asarray(weights).squeeze()

def stocGradAscent(dataSet,label):
    dataMat = np.mat(dataSet)
    m,n = np.shape(dataMat)
    lableMat = np.mat(label).transpose()

    weights = np.ones(n)

    maxCyclye = 800

    for i in range(maxCyclye):
        dataIndex = list(range(m))
        # print("len: ",len(dataIndex))
        for j in range(m):
            alpha = 4/(1+i+j) + 0.001
            randIdx = np.random.randint(len(dataIndex))
            # print("dataMat.shape: ",dataMat[dataIndex[randIdx]].shape)
            # print("weights.shape: ",weights.shape)
            h = sigmod(sum(np.asarray(dataMat[dataIndex[randIdx]]).squeeze()*weights))
            # print("cur: ",dataIndex[randIdx])
            error = lableMat[dataIndex[randIdx]] - h
            # print(error.shape)
            weights = weights + alpha * error * dataMat[dataIndex[randIdx]]
            weights = np.asarray(weights).squeeze()
            # print("wei.shape: ",weights.shape)
            del dataIndex[randIdx]
            # print("result: ",len(dataIndex))

    return weights

def plotBestFit(dataSet, label, weights):
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    m = len(dataSet)
    n = len([dataSet[0]])

    for i in range(m):
        if (label[i]==0):
            xcord1.append(dataSet[i][1])
            ycord1.append(dataSet[i][2])
        elif label[i]==1:
            xcord2.append(dataSet[i][1])
            ycord2.append(dataSet[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord1, ycord1, s= 30, c = 'blue', marker = 's')    ## 两种点分开画
    ax.scatter(xcord2, ycord2, s= 30, c = 'red',)

    x = np.arange(-4.0,4.0,0.10)
    # print("------weight:-----", weights)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)                                         # 画拟合直线

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def test(x_test, y_test, weights):
    count = 0
    total = len(x_test)
    for i in range(total):
        result = classifyVector(x_test[i], weights)
        if (abs(result - y_test[i]) < 0.000001):    # 浮点数判断相等要注意细节
            count += 1
    return count/total

def train():
    x_train, y_train, x_test, y_test = loadDataSet()
    # print(x_train, y_train)
    weights = stocGradAscent(x_train, y_train)
    # weights = gradAscent(dataSet, lable)
    plotBestFit(x_train, y_train, weights)
    # print(x_test, y_test)
    acc = test(x_test, y_test, weights)
    print('------acc------\n')
    print('finnal acc in test:', acc)

def classifyVector(x_test, weights):
    x = x_test[1]
    y = x_test[2]
    pred = (-weights[0]-weights[1]*x)/weights[2]
    # 考虑点在线的上方还是下方返回预测结果
    if (pred > y):
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    train()