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

import time

from abc import ABC, abstractmethod
from keras.models import Sequential

from callbacks.accuracy_history import AccuracyHistory


class AbstractModel(ABC):

    def __init__(self):
        self._model = None
        self._batch_size = 64
        self._epochs = 20
        self._verbose = 0
        self._history = AccuracyHistory()
        self._run_timestamp = str(time.time()).split('.')[0]

    @abstractmethod
    def create_model(self, input_shape):
        pass

    @abstractmethod
    def get_id(self) -> str:
        pass

    def get_run_id(self) -> str:
        return self.get_id() + '_' + self._run_timestamp

    def fit(self, features_train, labels_train, features_validation, labels_validation):
        model = self.get_model()
        model.fit(features_train, labels_train,
                  batch_size=self._batch_size,
                  epochs=self._epochs,
                  callbacks=[self._history],
                  validation_data=(features_validation, labels_validation),
                  verbose=self._verbose)

    def evaluate(self, features_validation, labels_validation):
        model = self.get_model()
        return model.evaluate(features_validation, labels_validation, verbose=self._verbose)

    def predict_classes(self, features_test):
        model = self.get_model()
        return model.predict_classes(features_test)

    def get_optimizer(self):
        return 'adam'

    def get_model(self) -> Sequential:
        if self._model is None:
            raise RuntimeError('You have not created the model yet. Call the `create_model` '
                               'method first!')

        return self._model

    def get_history(self):
        return self._history

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_verbose(self, verbose):
        self._verbose = verbose

    def _set_model(self, model: Sequential):
        self._model = model
