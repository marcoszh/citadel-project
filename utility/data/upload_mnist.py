import requests
import os
import io

from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib

from cryptography.fernet import Fernet
from ..native_encryption import NativeEncryptor

OLD_MODEL_KEY = b''
MODEL_KEY = b""

encryptor_m = NativeEncryptor(MODEL_KEY)

STORAGE_SVC_NAME = os.getenv("STORAGE_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")
STORAGE_URL = f'http://{STORAGE_SVC_NAME}.{SECUREML_NAMESPACE}/'


def upload_model_mongo(model):
    out_file = io.BytesIO()
    model.save(out_file)
    data = encryptor_m.encrypt(out_file.getvalue())
    requests.post(STORAGE_URL + 'upload_model/' + 'mnist_000', data=data,
                  headers={'Content-Type': 'application/octet-stream'})


tf.enable_eager_execution()
tfe = contrib.eager

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', input_shape=(28, 28, 1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))
model.summary()

upload_model_mongo(model)