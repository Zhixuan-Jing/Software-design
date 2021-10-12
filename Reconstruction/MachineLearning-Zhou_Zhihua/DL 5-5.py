'''
Machine Learning 5-5 
NN with 1 Hidden layer and BP
'''
import torch
import numpy as np


def loss(yH, y):
    return yH - y
    # Gradient of loss function


def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))
    return y


def test(W, V, test_set):
    answer = []
    for case in test_set:
        tes = torch.tensor(case)
        hidden = torch.matmul(W, tes)
        for attr in hidden:
            attr = sigmoid(attr)  # sigmoid(case^W), output of hidden layer
        output = torch.matmul(hidden, torch.t(V))
        output = sigmoid(output)
        answer.append(output)
    return answer


def train(W, V, train_set):
    print("training start...")
    a = 0.5
    for data in train_set:
        # feed forward
        label = data[2]
        data = torch.tensor([data[0], data[1]])
        hidden = torch.sigmoid(torch.matmul(W, data))
        output = torch.sigmoid(torch.matmul(hidden, torch.t(V)))
        # back propagation
        diff = output - label
        grad_v = output * (1 - output) * diff
        for i in range(3):
            for j in range(2):
                grad_w = grad_v * hidden[i] * (1 - hidden[i])
                W[i][j] = W[i][j] - a * grad_w * data[j]
        # grad_w=torch.matmul(grad_v,torch.matmul(torch.t(data),hidden*(1-hidden)))
        V = V - a * grad_v * torch.t(hidden)

    print("training finished")


if __name__ == "__main__":
    W = torch.rand(3, 2)
    V = torch.rand(1, 3)
    train_set = [
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.719, 0.103, 0]
    ]
    test_set = [
        [0.437, 0.211],  # should be 1
        [0.593, 0.042]  # should be 0
    ]
    train(W, V, train_set)  # training
    print(test(W, V, test_set))  # test the output
