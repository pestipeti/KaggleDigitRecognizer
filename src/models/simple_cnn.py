# MIT License
#
# Copyright (c) 2018 Peter Pesti <pestipeti@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from .abstract_model import AbstractModel


class SimpleCnnModel(AbstractModel):

    def __init__(self):
        super(SimpleCnnModel, self).__init__()

    def get_id(self):
        return 'cnn_6'

    def create_model(self, input_shape):
        km = Sequential()

        km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
        km.add(MaxPooling2D(pool_size=(2, 2)))
        km.add(Dropout(0.25))

        km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
        km.add(MaxPooling2D(pool_size=(2, 2)))
        km.add(Dropout(0.25))

        km.add(Flatten())
        km.add(Dense(units=128, activation='relu'))
        km.add(Dropout(0.25))

        km.add(Dense(units=64, activation='relu'))
        km.add(Dropout(0.25))

        km.add(Dense(units=10, activation='softmax'))
        km.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self._set_model(km)
