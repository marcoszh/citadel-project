import io
import os
import threading
import time
from tempfile import TemporaryFile

import h5py
import requests
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from flask import Flask, json
from flask import request, Response
import time
from logging.config import dictConfig

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib

import tempfile
from tempfile import TemporaryFile

# from cryptography.fernet import Fernet
from ..utility.native_encryption import NativeEncryptor

tmpdir = tempfile.mkdtemp()

tf.enable_eager_execution()
tfe = contrib.eager
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
STORAGE_SVC_NAME = os.getenv("STORAGE_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")
FRONTEND_SVC_NAME = os.getenv("FRONTEND_SVC_NAME")
MODEL_KEY = b"needstobesixteen"

encryptor_m = NativeEncryptor(MODEL_KEY)
# encryptor_m = Fernet(MODEL_KEY)

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'formatter': 'default'
    }},
    'root': {
        'level': 'DEBUG',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
WORKER_ID = 0

NEXT_ITERATION_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/start_iteration/'
STORAGE_URL = f'http://{STORAGE_SVC_NAME}.{SECUREML_NAMESPACE}/'

cce = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
global_step = tf.Variable(0)

global_model = None
gradient_list = []
start_time = None
started_lock = False

thread_lock = threading.Lock()

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
gradient_container_client = blob_service_client.get_container_client('gradients')
model_container_client = blob_service_client.get_container_client('model')
# Create the container
# container_client = blob_service_client.create_container(container_name)

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


def download_model_mongo(file_name, encryptor):
    response = requests.get(STORAGE_URL+'download_model/'+file_name)
    bio = io.BytesIO(encryptor.decrypt(response.content))
    bio = h5py.File(bio, mode='r')
    return bio


def download_decrypt_gradients_mongo(iter_num, worker_id, encryptor):
    response = requests.get(STORAGE_URL+'download_gradients/'+'gradients_{:03d}_{:03d}'.format(worker_id, iter_num))
    grads = uncompress_nparr(encryptor.decrypt(response.content))
    return grads


@app.route('/upload_gradient/<worker_id>', methods=['POST'])
def upload_gradient(worker_id):
    global gradient_list, thread_lock, start_time, started_lock
    app.logger.info("start receiving grads. worker: {:03d}".format(int(worker_id)))
    grads = uncompress_nparr(encryptor_m.decrypt(request.data))
    app.logger.info("grads decrypted. worker: {:03d}".format(int(worker_id)))
    thread_lock.acquire()
    gradient_list.append(grads)
    if started_lock == False:
        start_time = time.time()
        started_lock = True
    thread_lock.release()
    app.logger.info("grads saved. worker: {:03d}".format(int(worker_id)))

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

    sum_grads = np.sum(grad_list, axis=0)

    optimizer.apply_gradients(zip(sum_grads, global_model.trainable_variables), global_step)

    out_file = io.BytesIO()
    global_model.save(out_file)
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    upload_blob('model', encrypted_model, 'mnist_{:03d}'.format(iter_num+1))
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
    upload_blob('model', encrypted_model, 'mnist_{:03d}'.format(iter_num+1))
    # bytestream = bytestream_nparr(grads)
    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)

@app.route('/finish_iteration/<iter_num>', methods=['POST'])
def finish_iteration(iter_num):
    global global_model

    start_t = time.time()
    app.logger.info("start to finish aggregation. iter_num: {:03d}".format(int(iter_num)))

    app.logger.info("start decrypting grads. iter_num: {:03d}".format(int(iter_num)))
    grads = uncompress_nparr(encryptor_m.decrypt(request.data))
    app.logger.info("grads decrypted. iter_num: {:03d}".format(int(iter_num)))
    
    app.logger.info("start applying grads. iter_num: {:03d}".format(int(iter_num)))
    optimizer.apply_gradients(
        zip(grads, global_model.trainable_variables), global_step)
    app.logger.info("grads applied. iter_num: {:03d}".format(int(iter_num)))
    
    app.logger.info("start serialize model. iter_num: {:03d}".format(int(iter_num)))
    out_file = io.BytesIO()
    global_model.save(out_file)
    app.logger.info("model serialized. iter_num: {:03d}".format(int(iter_num)))

    app.logger.info("start encrypting model. iter_num: {:03d}".format(int(iter_num)))
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    app.logger.info("model encrypted. iter_num: {:03d}".format(int(iter_num)))

    app.logger.info("start uploading model. iter_num: {:03d}".format(int(iter_num)))
    upload_model_mongo('mnist_{:03d}'.format(int(iter_num)+1), encrypted_model)
    app.logger.info("model uploaded. iter_num: {:03d}".format(int(iter_num)))

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))

    app.logger.info("start informing frontend to start next iteration. iter_num: {:03d}".format(int(iter_num)))
    response = requests.get(NEXT_ITERATION_URL + str(int(iter_num)+1))
    app.logger.info("inform done. iter_num: {:03d}".format(int(iter_num)))

    return Response(status=200)

@app.route('/aggregate_mongo/<prefix>', methods=['POST'])
def aggregate_mongo(prefix):
    global global_model, thread_lock, gradient_list, start_time, started_lock
    # start_t = time.time()
    # getting masks matching prefix
    iter_num = int(request.json['iter_num'])
    app.logger.info("starting aggregation. iter_num: {:03d}".format(iter_num))

    thread_lock.acquire()
    sum_grads = np.sum(gradient_list, axis=0)
    gradient_list = []
    thread_lock.release()
    app.logger.info("grads aggregated. iter_num: {:03d}".format(iter_num))

    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    optimizer.apply_gradients(
        zip(sum_grads, global_model.trainable_variables), global_step)
    app.logger.info("grads applied. iter_num: {:03d}".format(iter_num))
    # out_file = TemporaryFile()
    out_file = io.BytesIO()
    global_model.save(out_file)
    app.logger.info("model serialized. iter_num: {:03d}".format(iter_num))
    encrypted_model = encryptor_m.encrypt(out_file.getvalue())
    app.logger.info("model encrypted. iter_num: {:03d}".format(iter_num))
    upload_model_mongo('mnist_{:03d}'.format(iter_num+1), encrypted_model)
    app.logger.info("model uploaded. iter_num: {:03d}".format(iter_num))
    elapsed_time = time.time() - start_time
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    started_lock = False
    return Response(status=200)


if __name__ == "__main__":
    # print('lala')
    # threading.Thread(target=app.run).start()
    global_model = download_model_mongo('mnist_000', encryptor_m)
    global_model = keras.models.load_model(global_model)
    app.run(debug=False, port=80, host='0.0.0.0', threaded=True)
