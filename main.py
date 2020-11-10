import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(file_name):
    dataMat = []
    labelMat = []

    with open(file_name) as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))

    return dataMat,labelMat

def sigmod(Z):
    return 1.0 / (1+ np.exp(-Z))

def thanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def gradAscent(dataSet,label):
    dataMat = np.mat(dataSet)   #  (m,n)
    m,n = np.shape(dataMat)
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
    alpha = 0.001

    maxCyclye = 800

    for i in range(maxCyclye):
        dataIndex = list(range(m))
        #print("len: ",len(dataIndex))
        for j in range(m):
            alpha = 4/(1+i+j) + 0.001
            randIdx = np.random.randint(len(dataIndex))
            #print("dataMat.shape: ",dataMat[dataIndex[randIdx]].shape)
            #print("weights.shape: ",weights.shape)
            h = sigmod(sum(np.asarray(dataMat[dataIndex[randIdx]]).squeeze()*weights))
            #print("cur: ",dataIndex[randIdx])
            error = lableMat[dataIndex[randIdx]] - h
            #print(error.shape)
            weights = weights + alpha * error * dataMat[dataIndex[randIdx]]
            weights = np.asarray(weights).squeeze()
            #print("wei.shape: ",weights.shape)
            del dataIndex[randIdx]
            #print("result: ",len(dataIndex))

    return weights

def plotBestFit(dataSet,label,weights):

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

    ax.scatter(xcord1, ycord1, s= 30, c = 'red',marker = 's')    ## 两种点分开画
    ax.scatter(xcord2, ycord2, s= 30, c = 'green',)

    x = np.arange(-2.0,2.0,0.10)
    print("------------weight:-----",weights)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)                                         #  画拟合直线

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def simpleTest():
    dataSet, lable = loadDataSet('horse.txt')
    weights = stocGradAscent(dataSet,lable)
    #weights = gradAscent(dataSet,lable)
    plotBestFit(dataSet,lable,weights)

def classifyVector(intX,weights):
    '''
        Desc:
            最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
        Args:
            inX -- 特征向量，features
            weights -- 根据梯度下降/随机梯度下降 计算得到的回归系数
        Returns:
            如果 prob 计算大于 0.5 函数返回 1
            否则返回 0
    '''

    prob = sigmod(intX*weights)
    if prob>0.5:
        return 1.0
    else:return 0.0


if __name__ == '__main__':
    data,lable = loadDataSet('horse.txt')
    print("data:", data, '\n')
    print("lable:", lable, '\n')
    # print(type(lable))
    # m = np.mat(data)
    # print(m)
    # print(type(m))
    # l = np.mat(lable)
    # print(l)
    # print(l.shape)
    # print(l.transpose())
    # print(np.exp(a))
    # a = np.array([1])
    simpleTest()