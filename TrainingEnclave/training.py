import io
import time
import os
import uuid

import zlib

import h5py
import requests
from flask import Flask, json
from flask import request, Response
import time
import atexit

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib
from logging.config import dictConfig
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
from tempfile import TemporaryFile

import threading

from queue import Queue

from ..utility.aggr_tree_processor_center import AggrTreeProcessorCenter

# from cryptography.fernet import Fernet
from ..utility.native_encryption import NativeEncryptor

tmpdir = tempfile.mkdtemp()

tf.enable_eager_execution()
tfe = contrib.eager

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

# For k8s deployment
POD_IP = os.getenv("KUBE_POD_IP")
FRONTEND_SVC_NAME = os.getenv("FRONTEND_SVC_NAME")
AGGREGATOR_SVC_NAME = os.getenv("AGGREGATOR_SVC_NAME")
STORAGE_SVC_NAME = os.getenv("STORAGE_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")
NUM_OF_WORKERS = os.getenv("NUM_OF_WORKERS")
AGGR_TREE_BATCH = os.getenv("AGGR_TREE_BATCH")

REG_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/register_worker/'
UNREG_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/unregister_worker/'
MASK_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/req_mask/'
AGGREGATE_URL = f'http://{AGGREGATOR_SVC_NAME}.{SECUREML_NAMESPACE}/upload_gradient/'
AGGREGATOR_NEXT_ITER_URL = f'http://{AGGREGATOR_SVC_NAME}.{SECUREML_NAMESPACE}/finish_iteration/'
STORAGE_URL = f'http://{STORAGE_SVC_NAME}.{SECUREML_NAMESPACE}/'

# MASK_URL = 'http://docker.for.mac.localhost:5001/req_mask/'
# AGGREGATE_URL = 'http://docker.for.mac.localhost:5002/upload_gradient/'

LEARNING_RATE = 0.001
BATCH_SIZE = 32
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODEL_KEY = b''
encryptor_m = NativeEncryptor(MODEL_KEY)
# encryptor_m = Fernet(MODEL_KEY)

queue_lock = threading.Lock()
post_queue = Queue()

cce = keras.losses.SparseCategoricalCrossentropy()

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
# container_name = "quickstart" + str(uuid.uuid4())
# Create the container
# container_client = blob_service_client.create_container(container_name)

global_worker_id = -1

aggr_tree_center = None


def aggr_tree_next_iter(params):
    global global_worker_id
    worker_id = global_worker_id

    iter_num = params["iter_num"]
    data = params["data"]
    level = params["level"]
    num = params["num"]

    app.logger.info("push up data to aggregator out of aggregation tree. worker: {:03d}, iter_num: {:03d}, level: {}, num: {}".
                        format(worker_id, iter_num, level, num))
    # threading.Thread(target=requests.post, kwargs={
    #     "url": AGGREGATOR_NEXT_ITER_URL + str(iter_num),
    #     "data": data,
    #     "headers": {'Content-Type': 'application/octet-stream'}
    # }).start()
    requests.post(url=AGGREGATOR_NEXT_ITER_URL + str(iter_num), data=data, headers={'Content-Type': 'application/octet-stream'})

def aggr_tree_pushup(params):
    global global_worker_id
    worker_id = global_worker_id

    iter_num = params["iter_num"]
    data = params["data"]
    level = params["level"]
    num = params["num"]
    ip = params["ip"]

    app.logger.info("push up data to upper level in aggregation tree. worker: {:03d}, iter_num: {:03d}, level: {}, num: {}".
                        format(worker_id, iter_num, level, num))
    # threading.Thread(target=requests.post, kwargs={
    #     "url": "http://" + ip + "/aggr_tree_receive_data/{}/{}/{}".format(iter_num, level, num),
    #     "data": data,
    #     "headers": {'Content-Type': 'application/octet-stream'}
    # }).start()
    requests.post(url="http://" + ip + "/aggr_tree_receive_data/{}/{}/{}".format(iter_num, level, num), 
                    data=data, headers={'Content-Type': 'application/octet-stream'})

def _loss(_model, x, y):
    y_ = _model(x)
    return cce(y_true=y, y_pred=y_)


def _grad(_model, inputs, targets):
    with tf.GradientTape() as tape:
        _loss_value = _loss(_model, inputs, targets)
    return _loss_value, tape.gradient(_loss_value, _model.trainable_variables)


# def compress_nparr(nparr):
#     """
#     Returns the given numpy array as compressed bytestring,
#     the uncompressed and the compressed byte size.
#     """
#     bytestream = io.BytesIO()
#     np.save(bytestream, nparr)
#     uncompressed = bytestream.getvalue()
#     compressed = zlib.compress(uncompressed)
#     return compressed, len(uncompressed), len(compressed)

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


def download_decrypt_model(container_name, blob_name, local_name, encryptor):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    bio = io.BytesIO(encryptor.decrypt(blob_client.download_blob().readall()))
    return h5py.File(bio, mode='r')


def upload_blob_file(container_name, local_file, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    local_file.seek(0)
    blob_client.upload_blob(local_file)


def upload_blob(container_name, data: bytes, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data)


def download_decrypt_model_mongo(file_name, encryptor):
    response = requests.get(STORAGE_URL+'download_model/'+file_name)
    bio = io.BytesIO(encryptor.decrypt(response.content))
    return h5py.File(bio, mode='r')


def download_decrypt_mask_mongo(file_name, encryptor):
    response = requests.get(STORAGE_URL+'download_mask/'+file_name)
    mask = uncompress_nparr(encryptor.decrypt(response.content))
    return mask


def download_decrypt_data_mongo(file_name, encryptor):
    response = requests.get(STORAGE_URL+'download_data/'+file_name)
    data_ = uncompress_nparr(encryptor.decrypt(response.content))
    train_x = data_['x']
    train_y = data_['y']
    return train_x, train_y

@app.route('/generate_gradient/<worker_id>', methods=['POST'])
def generate_gradient(worker_id):
    start_t = time.time()
    worker_id = int(worker_id)
    iter_num = int(request.json['iter_num'])
    model_file = download_decrypt_model('model', 'mnist_{0:03d}'.format(iter_num), 'mnist_{0:03d}'.format(iter_num),
                                        encryptor_m)
    model = keras.models.load_model(model_file)
    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    train_y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
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
    upload_blob('gradients', encrypted_grads, 'masked_grads_{0:03d}_{1:02d}'.format(iter_num, worker_id))

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)


@app.route('/generate_gradient_upload/<worker_id>', methods=['POST'])
def generate_gradient_upload(worker_id):
    start_t = time.time()
    worker_id = int(worker_id)
    iter_num = int(request.json['iter_num'])
    model_file = download_decrypt_model('model', 'mnist_{0:03d}'.format(iter_num), 'mnist_{0:03d}'.format(iter_num),
                                        encryptor_m)
    model = keras.models.load_model(model_file)
    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    train_y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
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
    requests.post(url=AGGREGATE_URL + str(worker_id), data=encrypted_grads,
                  headers={'Content-Type': 'application/octet-stream'})

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)

def exec_aggr_tree_receive_data(params):
    global aggr_tree_center, global_worker_id
    iter_num = int(params["iter_num"])
    data = params["data"]
    level_from = int(params["level"])
    num_from = int(params["num"])
    app.logger.info("start to process aggr tree data from level: {}, num: {}. workerid: {}, iter_num:{}".
                    format(level_from, num_from, global_worker_id, iter_num))
    aggr_tree_center.receive_data(iter_num, data, level_from, num_from)
    app.logger.info("finish to process aggr tree data from level: {}, num: {}. workerid: {}, iter_num:{}".
                format(level_from, num_from, global_worker_id, iter_num))


@app.route('/aggr_tree_receive_data/<iter_num>/<level_from>/<num_from>', methods=['POST'])
def aggr_tree_receive_data(iter_num, level_from, num_from):
    exec_aggr_tree_receive_data({
        "iter_num": int(iter_num),
        "data": request.data,
        "level": int(level_from),
        "num": int(num_from),
    })
    return Response(status=200)

@app.route('/start_iteration/<iter_num>', methods=['POST', 'GET'])
def start_iteration(iter_num):
    global global_worker_id
    worker_id = int(global_worker_id)
    iter_num = int(iter_num)

    start_t = time.time()
    app.logger.info("starting training enclave. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    app.logger.info("start downloading model. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    model_file = download_decrypt_model_mongo(
    'mnist_{0:03d}'.format(iter_num), encryptor_m)
    app.logger.info("model downloaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    app.logger.info("start loading model. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    model = keras.models.load_model(model_file)
    app.logger.info("model loaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    app.logger.info("start downloading data. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    train_x, train_y = download_decrypt_data_mongo('data_{:03d}_{:03d}'.format(worker_id, iter_num), encryptor_m)
    app.logger.info("data downloaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    app.logger.info("start generating grads. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    loss_value, grads = _grad(model, train_x, train_y)
    app.logger.info("grads generated. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    app.logger.info("start converting grads. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    grads = [g.numpy() for g in grads]
    app.logger.info("worker {}, {}".format(worker_id, grads[0].flatten()[0]))
    app.logger.info("grads converted. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
  
    app.logger.info("start encrypting grads. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    np_bytes = io.BytesIO()
    np.save(np_bytes, grads)
    encrypted_grads = encryptor_m.encrypt(np_bytes.getvalue())
    app.logger.info("grads encrypted. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
  
    app.logger.info("feeding data to aggregating tree. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    # threading.Thread(target=exec_aggr_tree_receive_data, args={
    #     "iter_num": iter_num,
    #     "data": encrypted_grads,
    #     "level": -1,
    #     "num": worker_id
    # }).start()
    exec_aggr_tree_receive_data({
        "iter_num": iter_num,
        "data": encrypted_grads,
        "level": -1,
        "num": worker_id
    })

    elapsed_time = time.time() - start_t
    app.logger.debug('training enclave finished in {:4f}s. iter_num: {:03d}'.format(elapsed_time, iter_num))
  
    return Response(status=200)


@app.route('/generate_gradient_mongo/<worker_id>', methods=['POST'])
def generate_gradient_mongo(worker_id):
    start_t = time.time()
    worker_id = int(worker_id)
    iter_num = int(request.json['iter_num'])
    app.logger.info("starting training enclave. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    model_file = download_decrypt_model_mongo(
        'mnist_{0:03d}'.format(iter_num), encryptor_m)
    app.logger.info("model downloaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    model = keras.models.load_model(model_file)
    app.logger.info("model loaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    # optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    # global_step = tf.Variable(0)
    # train_x = tf.random.uniform((BATCH_SIZE, 32, 32, 3), minval=0, maxval=1)
    # train_y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=10, dtype=tf.dtypes.int32)
    train_x, train_y = download_decrypt_data_mongo('data_{:03d}_{:03d}'.format(worker_id, iter_num), encryptor_m)
    app.logger.info("data downloaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    loss_value, grads = _grad(model, train_x, train_y)
    app.logger.info("grads generated. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    grads = [g.values.numpy() if isinstance(g, tf.IndexedSlices) else g.numpy() for g in grads]
    app.logger.info("grads converted. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    # request mask
    mask = download_decrypt_mask_mongo('mask_{:03d}_{:03d}'.format(worker_id, iter_num), encryptor_m)
    app.logger.info("mask downloaded: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    # for i, m in enumerate(mask):
    #     app.logger.info("{}, grad: {}, mask: {}".format(i, np.shape(grads[i]), np.shape(m)))
    # app.logger.info("grad: {}, mask: {}".format(np.shape(grads), np.shape(mask)))
    grads = np.sum([grads, mask], axis=0)
    app.logger.info("mask applied. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    # bytestream = bytestream_nparr(grads)
    np_bytes = io.BytesIO()
    # np.save(grad_file, grads)
    np.save(np_bytes, grads)
    encrypted_grads = encryptor_m.encrypt(np_bytes.getvalue())
    app.logger.info("masked grads encrypted. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))
    # requests.post(url=STORAGE_URL+'upload_gradients/'+'gradients_{:03d}_{:03d}'.format(worker_id, iter_num), data=encrypted_grads,
    #               headers={'Content-Type': 'application/octet-stream'})
    requests.post(url=AGGREGATE_URL + str(worker_id), data=encrypted_grads,
                  headers={'Content-Type': 'application/octet-stream'})
    app.logger.info("grads uploaded. worker: {:03d}, iter_num: {:03d}".format(worker_id, iter_num))

    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(status=200)

def get_np_value(x):
    y = io.BytesIO()
    np.save(y,x)
    return y.getvalue()

@app.route('/receive_ip_list', methods=['POST'])
def receive_ip_list():
    global aggr_tree_center, global_worker_id
    ip_list = request.json["ip_list"].split(",")
    app.logger.info("received ip list {}, worker_id: {}".format(ip_list, global_worker_id))
    # def getsum(x, y):
    #     result = np.sum([x, y], axis=0)
    #     app.logger.info("{} + {} = {}".format(x[0].flatten()[0], y[0].flatten()[0], result[0].flatten()[0]))
    #     return result
    aggr_tree_center = AggrTreeProcessorCenter(
        worker_num = int(NUM_OF_WORKERS),
        batch_size = int(AGGR_TREE_BATCH), 
        this_worker_id = int(global_worker_id),
        encrypt_func = lambda x: encryptor_m.encrypt(get_np_value(x)),
        decrypt_func = lambda x: uncompress_nparr(encryptor_m.decrypt(x)),
        aggr_func = lambda x, y: np.sum([x, y], axis=0),
        pushup_func = aggr_tree_pushup,
        next_iter_func = aggr_tree_next_iter,
        ip_list = ip_list,
        logger = app.logger
    )
    return Response(status=200)

def env_check():
    print("> POD_IP: {}".format(POD_IP))
    print("> FRONTEND_SVC_NAME: {}".format(FRONTEND_SVC_NAME))
    print("> AGGREGATOR_SVC_NAME: {}".format(AGGREGATOR_SVC_NAME))
    print("> STORAGE_SVC_NAME: {}".format(STORAGE_SVC_NAME))
    print("> SECUREML_NAMESPACE: {}".format(SECUREML_NAMESPACE))
    print("> NUM_OF_WORKERS: {}".format(NUM_OF_WORKERS))
    print("> AGGR_TREE_BATCH: {}".format(AGGR_TREE_BATCH))
    if POD_IP is None or FRONTEND_SVC_NAME is None or AGGREGATOR_SVC_NAME is None or SECUREML_NAMESPACE is None or STORAGE_SVC_NAME is None or NUM_OF_WORKERS is None or AGGR_TREE_BATCH is None:
        print("> There is missing env var, exit.")
        return False
    return True



def register_this_worker():
    global global_worker_id

    time.sleep(5)

    app.logger.info('> register this worker {}'.format(POD_IP))
    r = requests.get(REG_URL + POD_IP).json()
    app.logger.info('> register result: {}'.format(r))
    if "id" in r:
        worker_id = r["id"]
    if worker_id != -1:
        app.logger.info('> register succeeded, id: {}'.format(r["id"]))
        global_worker_id = worker_id

def unregister_this_worker():
    app.logger.info('> unregister this worker, ip: {}'.format(POD_IP))
    r = requests.get(UNREG_URL + POD_IP).json()
    app.logger.info('> unregister result: {}'.format(r))

def listen_post_event():
    app.logger.info("listening thread start")
    while True:
        if not post_queue.empty():
            queue_lock.acquire()
            req = post_queue.get()
            queue_lock.release()
            app.logger.info("detected post request {}".format(req))
            request.post(**req)

if __name__ == "__main__":
    if not env_check():
        exit(1)
    register_this_worker()
    atexit.register(unregister_this_worker)
    app.run(debug=False, port=80, host='0.0.0.0', threaded=True)
