# Training a regression model... on a classification problem! Nothing bad can happen.

import os
import numpy as np
from keras.datasets import mnist
from layers import *
from nn import NeuralNet
from tqdm import tqdm

def mnist_train_batch(batchsize):
    (train_img, train_label), (_, _) = mnist.load_data()
    # train_img.shape = (60000, 28, 28) -> (60000, 28*28)
    train_img = train_img.reshape(-1, 28*28)

    n_batches = 60000 // batchsize

    img_batches = np.split(train_img, n_batches)
    label_batches = np.split(train_label, n_batches)

    for i in range(n_batches):
        yield (img_batches[i], label_batches[i])

def mnist_test_iter():
    (_, _), (test_img, test_label) = mnist.load_data()
    test_img = test_img.reshape(-1, 28*28)

    for i in range(10000):
        yield (test_img[i], test_label[i])

evaluate = lambda result, expected: round(result) == expected
vevaluate = np.vectorize(evaluate)

if __name__ == '__main__':
    BATCHSIZE = 100

    # A: 28x28 -> 28
    # B: 28 -> 10
    # C: 10 -> 1
    A = MatrixLayer.init_random((28, 28*28))
    B = MatrixLayer.init_random((10, 28))
    C = VectorLayer.init_random(10)

    net = NeuralNet(A, Sigmoid(28), B, Sigmoid(10), C)

    # start training
    for i, (imgs, labels) in enumerate(mnist_train_batch(BATCHSIZE)):
        grads, net_error = net.compute_grad(imgs, labels)
        print("Batch %s, net error: %s" % (i, net_error))
        net.apply_grad(grads)
        net.clear()

    successes = 0
    for (img, label) in tqdm(mnist_test_iter()):
        res = net.apply(img)
        if evaluate(res, label): successes += 1

    print("%s successes out of 10000 (%s percent)" % (successes, successes / 100))
