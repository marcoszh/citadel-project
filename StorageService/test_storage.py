import io
import time

import h5py
import requests
import numpy as np
from cryptography.fernet import Fernet
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib
from ..utility import native_encryption

tf.enable_eager_execution()
tfe = contrib.eager

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_KEY = b'28Kg0oHex4c6y-DqXjo7Lr4Vmm5qlPZ3qQNdmaH1_w0='

encryptor_m = Fernet(MODEL_KEY)

cce = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
global_step = tf.Variable(0)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                              input_shape=(32, 32, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))
model.summary()

MODEL_URL = 'http://127.0.0.1:5536/'


def _loss(_model, x, y):
    y_ = _model(x)
    return cce(y_true=y, y_pred=y_)


def _grad(_model, inputs, targets):
    with tf.GradientTape() as tape:
        _loss_value = _loss(_model, inputs, targets)
    return _loss_value, tape.gradient(_loss_value, _model.trainable_variables)


upload = []
download = []

for batch_num in range(5):
    train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    train_y = tf.random.uniform(
        (BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
    loss_value, grads = _grad(model, train_x, train_y)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables), global_step)
    out_file = io.BytesIO()
    model.save(out_file)
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    # print(encrypted_model)
    start_t = time.time()
    requests.post(MODEL_URL+'upload_model/'+'alexnet_{:03d}'.format(
        batch_num+1), data=encrypted_model, headers={'Content-Type': 'application/octet-stream'})
    upload_t = time.time() - start_t
    time.sleep(2)
    start_t = time.time()
    response = requests.get(MODEL_URL+'download_model/' +
                            'alexnet_{:03d}'.format(batch_num+1))
    download_t = time.time() - start_t
    # print(response.content)
    bio = io.BytesIO(encryptor_m.decrypt(response.content))
    bio = h5py.File(bio, mode='r')
    model = keras.models.load_model(bio)
    upload.append(upload_t)
    download.append(download_t)
    print('upload: {:.4f}, download: {:.4f}'.format(upload_t, download_t))

print(np.mean(upload))
print(np.mean(download))
