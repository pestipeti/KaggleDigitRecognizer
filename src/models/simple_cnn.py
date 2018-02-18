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
from keras.preprocessing.image import ImageDataGenerator
from .abstract_model import AbstractModel


class SimpleCnnModel(AbstractModel):

    def __init__(self):
        super(SimpleCnnModel, self).__init__()

    def get_id(self):
        return 'data_aug_3'

    def create_model(self, input_shape):
        km = Sequential()

        km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
        km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
        km.add(MaxPooling2D(pool_size=(2, 2)))
        km.add(Dropout(0.25))

        km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
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

    def fit(self, features_train, labels_train, features_validation, labels_validation):
        generated_data = ImageDataGenerator(rotation_range=15,
                                            zoom_range=0.1,
                                            shear_range=0.1,
                                            height_shift_range=0.1,
                                            width_shift_range=0.1)

        generated_data.fit(features_train)

        model = self.get_model()
        model.fit_generator(generated_data.flow(features_train, labels_train,
                                                batch_size=self._batch_size),
                            epochs=self._epochs,
                            callbacks=[self._history],
                            validation_data=(features_validation, labels_validation),
                            steps_per_epoch=features_train.shape[0] / self._batch_size,
                            verbose=self._verbose)
