#!/usr/bin/env python

"""

"""

import arrow
import mlflow
import numpy as np
from pathlib import Path
import pickle
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


mlflow.set_experiment("MNIST with MLFlow")

image_size = 28
image_pixels = image_size * image_size
num_labels = 10

learning_rate = 0.05
loss_func = "categorical_crossentropy"
metrics = ["accuracy"]

mlflow.log_param('learning_rate', learning_rate)
mlflow.log_param('loss_function', loss_func)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, image_pixels)
x_test = x_test.reshape(-1, image_pixels)
y_train = np.eye(num_labels)[y_train].astype('int16')
y_test = np.eye(num_labels)[y_test].astype('int16')


def main():
    inputs = Input(shape=(image_pixels,))
    # hidden = Dense(int(image_pixels * 0.37), activation='relu')(inputs)
    outputs = Dense(num_labels, activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    sgd = SGD(lr=learning_rate)
    model.compile(optimizer=sgd, loss=loss_func, metrics=metrics)

    fit_resp = model.fit(x_train, y_train)
    mlflow.log_metric('training_loss', fit_resp.history.get('loss')[0])
    mlflow.log_metric('training_acc', fit_resp.history.get('acc')[0])

    loss, acc = model.evaluate(x_test, y_test)
    mlflow.log_metric('loss', loss)
    mlflow.log_metric('acc', acc)

if __name__ == '__main__':
    main()
