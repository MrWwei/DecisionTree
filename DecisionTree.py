#coding:utf-8
from math import log


class DecisionTree:
    trainData = []
    trainLabel = []
    featureValus = {}  # 每个特征值所有可能的取值

    def __init__(self, trainData, trainLabel, threshold):
        self.loadData(trainData, trainLabel)
        self.threshold = threshold
        self.tree = self.createTree(range(0, len(trainLabel)), range(0, len(trainData[0])))

    def loadData(self, trainData, trainLabel):
        if len(trainData) != len(trainLabel):
            raise ValueError('input error')
        self.trainData = trainData
        self.trainLabel = trainLabel

        for data in trainData:
            for index, value in enumerate(data):
                if not index in self.featureValus.keys():
                    self.featureValus[index] = [value]
                if not value in self.featureValus[index]:
                    self.featureValus[index].append(value)

    def caculateEntropy(self, dataset):
        labelCount = self.labelCount(dataset)
        size = len(dataset)
        result = 0
        for i in labelCount.values():
            pi = i / float(size)
            result -= pi * (log(pi) / log(2))
            return result

    def caculateGain(self, dataset, feature):
        values = self.featureValus[feature]
        result = 0
        for v in values:
            subDataset = self.splitDataset(dataset=dataset, feature=feature, value=v)
            result += len(subDataset) / float(len(dataset)) * self.caculateEntropy(subDataset)
        return self.caculateEntropy(dataset=dataset) - result

    def labelCount(self, dataset):
        labelCount = {}
        for i in dataset:
            if self.trainLabel[i] in labelCount.keys():
                labelCount[self.trainLabel[i]] += 1
            else:
                labelCount[self.trainLabel[i]] = 1
        return labelCount

    def createTree(self, dataset, features):
        labelCount = self.labelCount(dataset)
        if not features:
            return max(list(labelCount.items()), key=lambda x: x[1])[0]

        if len(labelCount) == 1:
            return labelCount.keys()[0]
        l = map(lambda x: [x, self.caculateGain(dataset=dataset, feature=x)], features)
        feature, gain = max(l, key=lambda x: x[1])
        if self.threshold > gain:
            return max(list(labelCount.items()), key=lambda x: x[1])[0]
        tree = {}
        subFeatures = filter(lambda x: x != feature, features)
        tree['feature'] = feature
        for value in self.featureValus[feature]:
            subDataset = self.splitDataset(dataset=dataset, feature=feature, value=value)
            if not subDataset:
                continue
            tree[value] = self.createTree(dataset=subDataset, features=subFeatures)
            return tree

    def splitDataset(self, dataset, feature, value):
        result = []
        for index in dataset:
            if self.trainData[index][feature] == value:
                result.append(index)
        return result

    def classify(self, data):
        def f(tree, data):
            if type(tree) != dict:
                return tree
            else:
                return f(tree[data[tree['feature']]], data)

        return f(self.tree, data)
if __name__ == '__main__':
    trainData = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1],
                 [1, 1, 1, 1], [1, 0, 1, 2], [1, 0, 1, 2], [2, 0, 1, 2], [2, 0, 1, 1], [2, 1, 0, 1], [2, 1, 0, 2],
                 [2, 0, 0, 0], ]
    trainLabel = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    tree = DecisionTree(trainData=trainData, trainLabel=trainLabel, threshold=0)
    print tree.tree