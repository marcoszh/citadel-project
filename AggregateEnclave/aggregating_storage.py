import io
import os
import threading
import time
from tempfile import TemporaryFile

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from flask import Flask, json
from flask import request, Response
import requests
import time
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib

import tempfile
from tempfile import TemporaryFile

from cryptography.fernet import Fernet

tmpdir = tempfile.mkdtemp()

tf.enable_eager_execution()
tfe = contrib.eager
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODEL_KEY = 

encryptor_m = Fernet(MODEL_KEY)

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MASK_URL = 'http://0.0.0.0/req_mask/'
STORAGE_URL = 'http://127.0.0.1:5536/'
WORKER_ID = 0

cce = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
global_step = tf.Variable(0)

global_model = None
gradient_list = []

thread_lock = threading.Lock()

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
gradient_container_client = blob_service_client.get_container_client('gradients')
model_container_client = blob_service_client.get_container_client('model')
# Create the container
# container_client = blob_service_client.create_container(container_name)

# Set the logging level for all azure-storage-* libraries
logger = logging.getLogger('azure.storage')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('azure')
logger.setLevel(logging.ERROR)


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
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # outfile = TemporaryFile()
    # outfile.write(blob_client.download_blob().readall())
    # return outfile
    local_name = tmpdir + '/' + local_name
    with open(local_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return local_name


def download_decrypt_blob(container_name, blob_name, local_name, encryptor):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # outfile = TemporaryFile()
    # outfile.write(blob_client.download_blob().readall())
    # return outfile
    local_name = tmpdir + '/' + local_name
    with open(local_name, "wb") as download_file:
        download_file.write(encryptor.decrypt(blob_client.download_blob().readall()))
    return local_name


def download_decrypt_gradients(container_name, blob_list, encryptor):
    def download_worker(_container_name, blob_name):
        blob_client = blob_service_client.get_blob_client(container=_container_name, blob=blob_name)
        bio = io.BytesIO()
        blob_client.download_blob().readinto(bio)
        return bio
    grad_streams = [download_worker(container_name, blob.name) for blob in blob_list]
    # grad_streams = Parallel(n_jobs=3, prefer="threads")(delayed(download_worker)(container_name, blob.name) for blob in blob_list)
    decrypted_s = [io.BytesIO(encryptor.decrypt(s.getvalue())) for s in grad_streams]
    return decrypted_s


def upload_blob_file(container_name, local_file, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    local_file.seek(0)
    blob_client.upload_blob(local_file)


def upload_blob(container_name, data: bytes, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data)


def upload_model_mongo(file_name, model):
    requests.post(STORAGE_URL+'upload_model/'+file_name, data=model,
                  headers={'Content-Type': 'application/octet-stream'})


@app.route('/upload_gradient/<worker_id>', methods=['POST'])
def upload_gradient(worker_id):
    global gradient_list, thread_lock
    grads = uncompress_nparr(encryptor_m.decrypt(request.data))
    thread_lock.acquire()
    gradient_list.append(grads)
    thread_lock.release()
    return Response(status=200)




@app.route('/aggregate/<prefix>', methods=['POST'])
def aggregate(prefix):
    global global_model
    start_t = time.time()
    # getting masks matching prefix
    iter_num = int(request.json['iter_num'])

    blob_list = gradient_container_client.list_blobs(name_starts_with='masked_grads_{0:03}_'.format(iter_num))
    grad_files = [download_decrypt_blob('gradients', blob.name, blob.name, encryptor_m) for blob in blob_list]
    grad_list = [np.load(grad_file, allow_pickle=True) for grad_file in grad_files]
    # for blob in blob_list:
        # grad_file = download_decrypt_blob('gradients', blob.name, blob.name, encryptor_m)
        # grad_file.seek(0)
        # grad = np.load(grad_file, allow_pickle=True)
        # app.logger.debug("loaded grads: " + grad_file)
        # grad_list.append(grad)
        # if not sum_grads:
        #     sum_grads = grad
        # else:
        #     for i, layer in enumerate(grad):
        #         sum_grads[i] += layer
    sum_grads = np.sum(grad_list, axis=0)

    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    optimizer.apply_gradients(zip(sum_grads, global_model.trainable_variables), global_step)
    # out_file = TemporaryFile()
    out_file = io.BytesIO()
    global_model.save(out_file)
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    upload_blob('model', encrypted_model, 'alexnet_{:03d}'.format(iter_num+1))
    # bytestream = bytestream_nparr(grads)
    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)


@app.route('/aggregate_inm/<prefix>', methods=['POST'])
def aggregate_inm(prefix):
    global global_model, thread_lock, gradient_list
    start_t = time.time()
    # getting masks matching prefix
    iter_num = int(request.json['iter_num'])

    thread_lock.acquire()
    sum_grads = np.sum(gradient_list, axis=0)
    gradient_list = []
    thread_lock.release()

    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    optimizer.apply_gradients(zip(sum_grads, global_model.trainable_variables), global_step)
    # out_file = TemporaryFile()
    out_file = io.BytesIO()
    global_model.save(out_file)
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    upload_model_mongo('alexnet_{:03d}'.format(iter_num+1), encrypted_model)
    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)


if __name__ == "__main__":
    # print('lala')
    # threading.Thread(target=app.run).start()
    global_model = download_decrypt_blob('model', 'alexnet_000', 'model_file', encryptor_m)
    global_model = keras.models.load_model(global_model)
    app.run(debug=False, port=5002, host='0.0.0.0')
