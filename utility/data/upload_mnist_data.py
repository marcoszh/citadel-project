import io
import os
import requests
from pathlib import Path
import numpy as np  # linear algebra
import pandas as pd
from cryptography.fernet import Fernet
from ..native_encryption import NativeEncryptor
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
import os
# for dirname, _, filenames in os.walk('/Users/marc/Documents/git/SecureML/utility/data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

file_path = Path(__file__).parent.absolute()

DATA_KEY = b''
# encryptor_m = Fernet(DATA_KEY)
encryptor_m = NativeEncryptor(DATA_KEY)

NUM_OF_WORKERS = os.getenv("NUM_OF_WORKERS")
UPLOAD_BATCH_NUM = os.getenv("UPLOAD_BATCH_NUM")
STORAGE_SVC_NAME = os.getenv("STORAGE_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")
BATCH_SIZE = 32

STORAGE_URL = f'http://{STORAGE_SVC_NAME}.{SECUREML_NAMESPACE}/'

def upload_data_mongo(file_name, model):
    requests.post(STORAGE_URL+'upload_data/'+file_name, data=model,
                headers={'Content-Type': 'application/octet-stream'})


# datagen = ImageDataGenerator(rescale=1./255,
#                              zoom_range=0.2,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              validation_split=0.2
#                              )
#
#
# train_data = datagen.flow_from_directory(os.path.join(file_path, 'input/gaussian_filtered_images/gaussian_filtered_images/'),
#                                          target_size=(96, 96),
#                                          batch_size=32,
#                                          class_mode='categorical',
#                                          subset='training')
#
#
# valid_data = datagen.flow_from_directory(os.path.join(file_path, 'input/gaussian_filtered_images/gaussian_filtered_images/'),
#                                          target_size=(96, 96),
#                                          batch_size=32,
#                                          class_mode='categorical',
#                                          subset='validation')


for worker_id in range(int(NUM_OF_WORKERS)):
    batch_id = 0
    while True:
        if batch_id == int(UPLOAD_BATCH_NUM):
            break
        print("uploading batch #{} for worker #{} ...".format(batch_id, worker_id))
        # y = [np.argmax(one_hot) for one_hot in y]
        # y = np.array(y).astype(np.int32)
        x = tf.random.uniform((BATCH_SIZE, 28, 28, 1), minval=0, maxval=1)
        y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
        bytestream = io.BytesIO()
        np.savez(bytestream, x=x.eval(session=tf.Session()), y=y.eval(session=tf.Session()))
        # data_ = np.load(io.BytesIO(bytestream.getvalue()), allow_pickle=True)
        # x = data_['x']
        # y = data_['y']
        encrypted_data = encryptor_m.encrypt(bytestream.getvalue())
        upload_data_mongo('data_{:03d}_{:03d}'.format(worker_id, batch_id), encrypted_data)
        batch_id += 1