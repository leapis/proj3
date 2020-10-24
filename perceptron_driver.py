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

def question_two(kRange, RUNS=100):
    m = 100
    epsilon = 0.05
    avg_steps = []

    for k in kRange:
        avg_steps.append(0)
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            avg_steps[-1] += p.steps
    
    return avg_steps

def question_three(eRange, RUNS=100):
    m = 100
    k = 5
    avg_steps = []

    for epsilon in eRange:
        avg_steps.append(0)
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            avg_steps[-1] += p.steps
    
    return avg_steps

###
# Bonus Questions
###

def bonus_diff(ws, bs):
    if len(ws) == 0:
        return -1
    norm = lambda x: np.array([0] * (len(x) - 1) + [1])
    w = ws[0]
    for i in range(1, len(ws)):
        w += ws[i]
    w /= len(ws)
    w = abs(w / sum(w))
    #print(w)
    w_2 = np.linalg.norm( norm(w) - w ) ** 2 
    b_2 = ((sum(bs)/len(bs))** 2)
    #print(w_2, b_2)
    return w_2 + b_2

def bonus_question_one(mRange, RUNS=1000):
    k = 5
    epsilon = 0.1
    bdiff = []

    for m in mRange:
        ws = []
        bs = []
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            ws.append(p.w)
            bs.append(p.b)
        bdiff.append(bonus_diff(ws, bs))
    
    return bdiff

def bonus_question_two(kRange, RUNS=1000):
    m = 100
    epsilon = 0.05
    bdiff = []

    for k in kRange:
        ws = []
        bs = []
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            ws.append(p.w)
            bs.append(p.b)
        bdiff.append(bonus_diff(ws, bs))
    
    return bdiff


def bonus_question_three(eRange, RUNS=1000):
    k = 5
    m = 100
    bdiff = []

    for epsilon in eRange:
        ws = []
        bs = []
        for _ in range(RUNS):
            x, y = generate_data(k, m, epsilon=epsilon)
            p = Perceptron(k)
            p.train(x, y)
            ws.append(p.w)
            bs.append(p.b)
        bdiff.append(bonus_diff(ws, bs))
    
    return bdiff


###
# main driver
###

if __name__ == "__main__":
    for i in bonus_question_one(np.arange(20, 201, 20)): print(i)

def testRun():
    k = 5
    m = 100
    percep = Perceptron(k)
    data, labels = generate_data(k,m)
    percep.train(data, labels)
    #print(percep.w, percep.b)