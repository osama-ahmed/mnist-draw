# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:10:08 2019

@author: Osama
"""

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_test=x_test/255

model=tf.keras.models.load_model('models/fashionmodel.h5')
model.summary()

w=28
h=28
x_test = x_test.reshape(x_test.shape[0], w, h, 1)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model.evaluate(x_test,y_test)
y_hat=model.predict(x_test[100].reshape(1,28,28,1))










model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('models/fashionweights.h5')
model.evaluate(x_test,y_test,verbose=0)

#model.save('fashionmodel.h5')
