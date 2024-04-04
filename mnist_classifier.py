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
    train_label = relabel_labels(train_label)

    n_batches = 60000 // batchsize

    img_batches = np.split(train_img, n_batches)
    label_batches = np.split(train_label, n_batches)

    for i in range(n_batches):
        yield (img_batches[i], label_batches[i])

def mnist_test_iter():
    (_, _), (test_img, test_label) = mnist.load_data()
    test_img = test_img.reshape(-1, 28*28)
    test_label = relabel_labels(test_label)

    for i in range(10000):
        yield (test_img[i], test_label[i])

def relabel_labels(mnist_labels):
    '''MNIST has the raw numbers as the labels; we want to have a unit vector with the n-th value as 1'''
    length = mnist_labels.shape[0] #this should be (n,)
    out = np.zeros((length, 10)) #10 different digits
    for i, n in enumerate(mnist_labels):
        out[i][n] = 1

    return out

def eval_binary(res, label):
    '''Given output vector and label vector, checks if the highest-probability category of the result is the label'''
    return label[np.argmax(res)] == 1

if __name__ == '__main__':
    BATCHSIZE = 50
    TRAIN_COUNT = 5
    STEP_RATE = (1/2)**(1/10)

    # A: 28x28 -> 28*14
    # B: 28*14 -> 14*14
    # C: 14*14 -> 7*7
    # C: 7*7 -> 1
    '''
    A = MatrixLayer.init_random((28*14, 28*28))
    B = MatrixLayer.init_random((14*14, 28*14))
    C = MatrixLayer.init_random((7*7, 14*14))
    D = VectorLayer.init_random(7*7)

    net = NeuralNet(A, Sigmoid(28*14), B, Sigmoid(14*14), C, Sigmoid(7*7), D)
    '''
    A = MatrixLayer.init_random((28, 28*28))
    B = MatrixLayer.init_random((10, 28))

    net = NeuralNet(A, ReLU(28), B, Sigmoid(10))
    # start training
    for j in range(TRAIN_COUNT):
        print('Training generation: %s' % (j+1))
        #steps = 0.01 * STEP_RATE**j
        for i, (imgs, labels) in enumerate(mnist_train_batch(BATCHSIZE)):
            grads, net_error = net.compute_grad_multi(imgs, labels, debug=True)
            print("Batch %s, net error: %s" % (i, net_error))
            #net.apply_grad(grads, steps)
            print('Gradients: %s' % grads)
            net.apply_grad(grads, 0.05)
            net.clear()
            input()

    successes = 0
    for (img, label) in mnist_test_iter():
        res = net.apply(img)
        print('Expected: %s' % label)
        print('Got: %s' % res)
        evec = res - label
        error = np.dot(evec, evec)
        print('Difference: %s' % error)
        if eval_binary(res, label): successes += 1
        input()

    print("%s successes out of 10000 (%s percent)" % (successes, successes / 100))
