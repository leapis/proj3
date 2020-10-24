import numpy as np

class Perceptron:

    def __init__(self, k):
        self.k = k
        self.w = np.zeros(k)
        self.b = 0
        self.steps = 0

    def classify(self, vector):
        return 1 if np.dot(self.w, vector) + self.b > 0 else -1
    
    def is_correct(self, vector, label):
        return True if self.classify(vector) == label else False 
    
    def some_is_incorrect(self, dataset, labels):
        for i, row in enumerate(dataset):
            if not self.is_correct(row, labels[i]):
                return True, i
        return False, -1

    def update(self, vector, label):
        self.w += vector * label
        self.b += label
        self.steps += 1

    def train_single_vector(self, vector, label):
        if not self.is_correct(vector, label):
            self.update(vector, label)
    
    def train(self, dataset, labels):
        unfinished, i = self.some_is_incorrect(dataset, labels)
        while unfinished:
            self.update(dataset[i], labels[i])
            unfinished, i = self.some_is_incorrect(dataset, labels)
