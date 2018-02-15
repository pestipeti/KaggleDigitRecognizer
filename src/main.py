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

import os
import time
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from models.simple_cnn import SimpleCnnModel


# ================================================
# Constants
# ================================================
EPOCHS = 20
VERBOSE = 1

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_DEPTH = 1

FOLDER_ROOT = './../'
FOLDER_INPUT = FOLDER_ROOT + '/input'
FOLDER_OUTPUT = FOLDER_ROOT + '/output'

start = time.time()
# ================================================
# Data preprocessing
# ================================================
train = pd.read_csv(FOLDER_INPUT + '/train.csv')
test = pd.read_csv(FOLDER_INPUT + '/test.csv')

features_train = train.iloc[:, 1:].values
features_test = test.values
labels_train = train.iloc[:, 0].values

# reshape the data
features_train = features_train.reshape(len(features_train), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
features_test = features_test.reshape(len(features_test), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

features_train /= 255.0
features_test /= 255.0

# Convert the labels.
labels_train = to_categorical(labels_train, num_classes=10)

# Splitting the training set to training and validation subset
features_train, features_validation, labels_train, labels_validation = train_test_split(
    features_train, labels_train, test_size=0.25, random_state=0)


# ================================================
# Create model
# ================================================
model = SimpleCnnModel()
model.set_verbose(VERBOSE)
model.set_epochs(EPOCHS)

model.create_model((IMG_WIDTH, IMG_HEIGHT, 1))
model.fit(features_train, labels_train, features_validation, labels_validation)

final_loss, final_acc = model.evaluate(features_validation, labels_validation)


# ================================================
# Saving results...
# ================================================
print("\nValidation Loss: {0:.6f},\nValidation Accuracy: {1:.6f}".format(
    final_loss, final_acc))

if not os.path.exists(FOLDER_OUTPUT + '/' + model.get_run_id()):
    os.makedirs(FOLDER_OUTPUT + '/' + model.get_run_id())

# ImageId, Label
image_ids = np.arange(features_test.shape[0])+1
labels = model.predict_classes(features_test)

df = pd.DataFrame({'ImageId': image_ids, 'Label': labels}, columns=['ImageId', 'Label'])
df.to_csv(path_or_buf=FOLDER_OUTPUT + '/' + model.get_run_id() + '/submission.csv', index=None,
          header=True)

history = model.get_history()

plt.plot(range(1, EPOCHS + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(FOLDER_OUTPUT + '/' + model.get_run_id() + '/accuracy.png')

np.savetxt(FOLDER_OUTPUT + '/' + model.get_run_id() + '/accuracy.dat', np.array(history.acc))

print('Total running time: ', time.time()-start)
