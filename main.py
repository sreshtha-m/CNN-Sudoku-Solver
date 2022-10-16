import copy
import keras
import numpy as np
from model import get_model
from data_preprocesses import get_data
from matplotlib import pyplot as plt

x_train, x_test, y_train, y_test = get_data('sudoku_copy.csv')

model = get_model()
adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=2)
model.save('sudoku.h5')

print(history.history.keys())


def norm(a):
    return (a / 9) - .5


def denorm(a):
    return (a + .5) * 9


def inference_sudoku(sample):
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''

    feat = copy.copy(sample)

    while (1):

        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if (mask.sum() == 0):
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred


def test_accuracy(feats, labels):
    correct = 0

    for i, feat in enumerate(feats):
        pred = inference_sudoku(feat)
        true = labels[i].reshape((9, 9)) + 1

        if (abs(true - pred)).sum() == 0:
            correct += 1
            accuracy = correct / feats.shape[0]

    print(f'Final Accuracy:{accuracy}')
    return correct / feats.shape[0]


test_accuracy(x_test[:50], y_test[:50])
