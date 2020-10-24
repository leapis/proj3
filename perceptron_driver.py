import numpy as np
from perceptron import Perceptron

def generate_data(k, m, epsilon = 0.2):
    # init vectors
    x = np.array([np.random.standard_normal(k) for _ in range(m)])
    y = np.zeros(m, dtype=int)

    # divide by l2 norm
    for i in range(m):
        x[i] /= np.linalg.norm(x[i])

    # remove vectors within epsilon
    not_in_gap = abs(x[:,k-1]) > epsilon
    x = x[not_in_gap]

    # assign labels
    for i in range(len(x)):
        y[i] = 1 if x[i][k - 1] >= 0 else -1

    return x,y

###
# functions for producting output data for HW
###

def question_one(mRange, RUNS=100):
    k = 5
    epsilon = 0.1
    avg_steps = []

    for m in mRange:
        avg_steps.append(0)
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            avg_steps[-1] += p.steps
        avg_steps[-1] /= RUNS
    
    return avg_steps

def question_two():
    None

def question_three():
    None


#main driver

if __name__ == "__main__":
    for i in question_one(range(1, 202, 2)): print(i)

def testRun():
    k = 5
    m = 100
    percep = Perceptron(k)
    data, labels = generate_data(k,m)
    percep.train(data, labels)
    print(percep.w, percep.b)