import io
import time
import os
import uuid

import h5py
import requests
from flask import Flask, json
from flask import request, Response
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib
import logging
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
from tempfile import TemporaryFile

from cryptography.fernet import Fernet

tmpdir = tempfile.mkdtemp()

tf.enable_eager_execution()
tfe = contrib.eager

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MASK_URL = 'http://127.0.0.1:5001/req_mask/'
AGGREGATE_URL = 'http://127.0.0.1:5002/upload_gradient/'
STORAGE_URL = 'http://127.0.0.1:5536/'
WORKER_ID = 0
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODEL_KEY = b''

encryptor_m = Fernet(MODEL_KEY)

cce = keras.losses.SparseCategoricalCrossentropy()

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
# container_name = "quickstart" + str(uuid.uuid4())
# Create the container
# container_client = blob_service_client.create_container(container_name)

# Set the logging level for all azure-storage-* libraries
logger = logging.getLogger('azure.storage')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('azure')
logger.setLevel(logging.ERROR)


def _loss(_model, x, y):
    y_ = _model(x)
    return cce(y_true=y, y_pred=y_)


def _grad(_model, inputs, targets):
    with tf.GradientTape() as tape:
        _loss_value = _loss(_model, inputs, targets)
    return _loss_value, tape.gradient(_loss_value, _model.trainable_variables)


def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(bytestring), allow_pickle=True)


def bytestream_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    return bytestream


def download_blob(container_name, blob_name, local_name):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)
    # outfile = TemporaryFile()
    # outfile.write(blob_client.download_blob().readall())
    # return outfile
    local_name = tmpdir + '/' + local_name
    with open(local_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return local_name


def download_decrypt_blob(container_name, blob_name, local_name, encryptor):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)
    # outfile = TemporaryFile()
    # outfile.write(blob_client.download_blob().readall())
    # return outfile
    local_name = tmpdir + '/' + local_name
    with open(local_name, "wb") as download_file:
        download_file.write(encryptor.decrypt(
            blob_client.download_blob().readall()))
    return local_name


def download_decrypt_model(container_name, blob_name, local_name, encryptor):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)
    bio = io.BytesIO(encryptor.decrypt(blob_client.download_blob().readall()))
    return h5py.File(bio, mode='r')


def upload_blob_file(container_name, local_file, blob_name):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)
    local_file.seek(0)
    blob_client.upload_blob(local_file)


def upload_blob(container_name, data: bytes, blob_name):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)
    blob_client.upload_blob(data)


def download_decrypt_model_mongo(file_name, encryptor):
    response = requests.get(STORAGE_URL+'download_model/'+file_name)
    bio = io.BytesIO(encryptor.decrypt(response.content))
    return h5py.File(bio, mode='r')


@app.route('/generate_gradient/<worker_id>', methods=['POST'])
def generate_gradient(worker_id):
    start_t = time.time()
    worker_id = int(worker_id)
    iter_num = int(request.json['iter_num'])
    model_file = download_decrypt_model('model', 'alexnet_{0:03d}'.format(iter_num), 'alexnet_{0:03d}'.format(iter_num),
                                        encryptor_m)
    model = keras.models.load_model(model_file)
    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    train_y = tf.random.uniform(
        (BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
    loss_value, grads = _grad(model, train_x, train_y)
    grads = [g.numpy() for g in grads]
    # request mask
    response = requests.get(MASK_URL + str(worker_id))
    masks = uncompress_nparr(response.content)
    # for i, mask in enumerate(masks):
    #     grads[i] += mask
    grads = np.sum([grads, masks], axis=0)

    # bytestream = bytestream_nparr(grads)
    np_bytes = io.BytesIO()
    # np.save(grad_file, grads)
    np.save(np_bytes, grads)
    encrypted_grads = encryptor_m.encrypt(np_bytes.getvalue())
    # grad_file = TemporaryFile()
    # grad_file.write(encrypted_grads)
    upload_blob('gradients', encrypted_grads,
                'masked_grads_{0:03d}_{1:02d}'.format(iter_num, worker_id))

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)


@app.route('/generate_gradient_upload/<worker_id>', methods=['POST'])
def generate_gradient_upload(worker_id):
    start_t = time.time()
    worker_id = int(worker_id)
    iter_num = int(request.json['iter_num'])
    model_file = download_decrypt_model_mongo(
        'alexnet_{0:03d}'.format(iter_num), encryptor_m)
    model = keras.models.load_model(model_file)
    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    # train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    # train_y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
    response = request.get(STORAGE_URL+'download_data/' +
                           'data_{:03d}_{:03d}'.format(worker_id, iter_num))
    data_ = uncompress_nparr(response.content)
    train_x = data_['x']
    train_y = data_['y']
    loss_value, grads = _grad(model, train_x, train_y)
    grads = [g.numpy() for g in grads]
    # request mask
    response = requests.get(MASK_URL + str(worker_id))
    masks = uncompress_nparr(response.content)
    # for i, mask in enumerate(masks):
    #     grads[i] += mask
    grads = np.sum([grads, masks], axis=0)

    # bytestream = bytestream_nparr(grads)
    np_bytes = io.BytesIO()
    # np.save(grad_file, grads)
    np.save(np_bytes, grads)
    encrypted_grads = encryptor_m.encrypt(np_bytes.getvalue())
    requests.post(url=AGGREGATE_URL+str(worker_id), data=encrypted_grads,
                  headers={'Content-Type': 'application/octet-stream'})

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)


if __name__ == "__main__":
    app.run(debug=False, port=5003, host='0.0.0.0')
