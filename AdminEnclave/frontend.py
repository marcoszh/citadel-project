import numpy as np
import zlib
import io
import os
import time

import requests
from flask import Flask
from flask import request, Response, jsonify
import threading
from cryptography.fernet import Fernet
from ..utility.native_encryption import NativeEncryptor

from logging.config import dictConfig

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

# For k8s deploy
NUM_OF_WORKERS = os.getenv("NUM_OF_WORKERS")
AGGREGATOR_SVC_NAME = os.getenv("AGGREGATOR_SVC_NAME")
STORAGE_SVC_NAME = os.getenv("STORAGE_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")

AGGREGATE_URL = f'http://{AGGREGATOR_SVC_NAME}.{SECUREML_NAMESPACE}/'
STORAGE_URL = f'http://{STORAGE_SVC_NAME}.{SECUREML_NAMESPACE}/'
# number of iterations to run
ITER_NUM = 10

# For worker management
worker_mgr_lock = threading.Lock()
worker_ip_to_index = dict()
worker_ip_list = dict()
worker_num = 0

MASK_SUM = None
MASK_SCALE = 100
MASKS = []
# SHAPES = [[3, 3, 3, 32], [32], [3, 3, 32, 32], [32], [3, 3, 32, 64], [64], [3, 3, 64, 64], [64], [30976, 512], [512],
 #           [512, 10], [10]]
SHAPES = [[5, 5, 1, 32], [32], [5, 5, 32, 32], [32], [3, 3, 32, 64], [64], [3, 3, 64, 64], [64], [3136, 256], [256], [256, 10], [10]]
MODEL_KEY = 
DATA_KEYS = {
    0: b"needstobesixteen",
    1: b"needstobesixteen",
    2: b"needstobesixteen",
    3: b"needstobesixteen",
}
DATA_BLOBS = {}
READY = False

thread_lock = threading.Lock()


def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)


def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


def bytestream_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    return bytestream


@app.route('/upload_data_key/<worker_id>', methods=['POST'])
def upload_data_key(worker_id):
    global DATA_KEYS
    start_t = time.time()
    if request.json['key']:
        DATA_KEYS[worker_id] = request.json['key']
        return Response(status=200)
    return Response(status=400)


@app.route('/upload_model_key', methods=['POST'])
def upload_model_key(worker_id):
    global MODEL_KEY, READY
    start_t = time.time()
    if request.json['key']:
        MODEL_KEY = request.json['key']
        READY = True
        return Response(status=200)
    return Response(status=400)


@app.route('/upload_shapes', methods=['POST'])
def upload_shapes():
    global SHAPES
    SHAPES = eval(request.json['shapes'])
    return Response(status=200)


@app.route('/prepare_offline_masks', methods=['POST', 'GET'])
def prepare_offline_masks(client_num):
    global SHAPES, MASKS, thread_lock
    start_t = time.time()
    client_num = int(client_num)
    app.logger.debug('Preparing {} masks'.format(client_num))
    thread_lock.acquire()
    MASKS = []
    for idx in range(client_num - 1):
        mask = [np.random.rand(*shape).astype(np.float16) * MASK_SCALE - 0.5 * MASK_SCALE for shape in SHAPES]
        MASKS.append(mask)
    MASKS.append((0 - np.sum(MASKS, axis=0)).tolist())
    thread_lock.release()
    elapsed_time = time.time() - start_t
    app.logger.debug('Prepared masks in {:4f}s'.format(elapsed_time))
    return Response(status=200)


@app.route('/req_mask/<worker_id>', methods=['POST', 'GET'])
def request_mask(worker_id):
    global MASKS, thread_lock
    start_t = time.time()
    thread_lock.acquire()
    mask = MASKS.pop(0)
    # app.logger.info('requested mask shape: {}'.format(np.shape(mask)))
    thread_lock.release()
    bytestream = bytestream_nparr(mask)
    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(response=bytestream.getvalue(), status=200,
                    mimetype="application/octet_stream")


@app.route('/register_worker/<worker_ip>', methods=['POST', 'GET'])
def register_worker(worker_ip):
    global worker_num

    app.logger.debug('Received register command from worker {}'.format(worker_ip))

    # get lock
    worker_mgr_lock.acquire()

    # existed worker
    if worker_ip in worker_ip_to_index:
        idx = worker_ip_to_index[worker_ip]
        app.logger.debug('Worker {} is already registered, id: {}'.format(worker_ip, idx))
        worker_mgr_lock.release()
        return jsonify({
            "msg": "existed",
            "id": idx
        })

    # register worker
    idx = -1
    for i in range(int(NUM_OF_WORKERS)):
        if i not in worker_ip_list:
            worker_ip_list[i] = worker_ip
            worker_ip_to_index[worker_ip] = i
            idx = i
            break

    # full
    if idx == -1:
        app.logger.debug('Worker {} register fail as the worker pool is full'.format(worker_ip))
        worker_mgr_lock.release()
        return jsonify({
            "msg": "full",
            "id": idx
        })

    # succeeded
    app.logger.debug('Worker {} register succeeded, id: {}'.format(worker_ip, idx))
    worker_num += 1
    worker_mgr_lock.release()
    return jsonify({
        "msg": "registered",
        "id": idx
    })


@app.route('/unregister_worker/<worker_ip>', methods=['POST', 'GET'])
def unregister_worker(worker_ip):
    global worker_num
    app.logger.debug('Received unregister command from worker {}'.format(worker_ip))

    # get lock
    worker_mgr_lock.acquire()

    # not exist
    if worker_ip not in worker_ip_to_index:
        app.logger.debug('Worker {} does not exist'.format(worker_ip))
        worker_mgr_lock.release()
        return jsonify({
            "msg": "not existed"
        })

    # unregister
    idx = worker_ip_to_index[worker_ip]
    del worker_ip_list[idx]
    del worker_ip_to_index[worker_ip]

    worker_mgr_lock.release()
    app.logger.debug('Worker {} unregistered'.format(worker_ip))
    worker_num -= 1
    return jsonify({
        "msg": "unregistered"
    })


def prepare_offline_masks_worker(client_num):
    global SHAPES, MASKS, thread_lock
    start_t = time.time()
    client_num = int(client_num)
    app.logger.debug('Preparing {} masks'.format(client_num))
    thread_lock.acquire()
    MASKS = []
    for idx in range(client_num - 1):
        mask = [np.random.rand(*shape).astype(np.float16) * MASK_SCALE - 0.5 * MASK_SCALE for shape in SHAPES]
        MASKS.append(mask)
    MASKS.append((0 - np.sum(MASKS, axis=0)).tolist())
    thread_lock.release()
    elapsed_time = time.time() - start_t
    app.logger.debug('Prepared masks in {:4f}s'.format(elapsed_time))


def upload_mask_mongo(iter_id, worker_id, mask):
    np_bytes = io.BytesIO()
    np.save(np_bytes, mask)
    # encrypted_masks = Fernet(DATA_KEYS[worker_id]).encrypt(np_bytes.getvalue())
    encrypted_masks = NativeEncryptor(DATA_KEYS[worker_id]).encrypt(np_bytes.getvalue())
    requests.post(url=STORAGE_URL + 'upload_mask/' + 'mask_{:03d}_{:03d}'.format(worker_id, iter_id),
                 data=encrypted_masks,
                 headers={'Content-Type': 'application/octet-stream'})


def prepare_offline_masks_mongo(client_num, iter_num):
    global SHAPES
    start_t = time.time()
    client_num = int(client_num)
    app.logger.debug('Preparing {} masks'.format(client_num))
    for iter in range(iter_num):
        mask_a = []
        for idx in range(client_num - 1):
            mask = [np.random.rand(*shape).astype(np.float16) * MASK_SCALE - 0.5 * MASK_SCALE for shape in SHAPES]
            mask_a.append(mask)
            upload_mask_mongo(iter, idx, mask)
        mask = (0 - np.sum(mask_a, axis=0)).tolist()
        upload_mask_mongo(iter, client_num - 1, mask)
        app.logger.debug('Uploading masks')
    elapsed_time = time.time() - start_t
    app.logger.debug('Prepared masks in {:4f}s'.format(elapsed_time))


def training_encalve_worker(url, worker_id, iter_num):
    response = requests.post("http://"+url + "/generate_gradient/" + str(worker_id), json={'iter_num': iter_num})
    return response.status_code


def training_thread(prefix, data_keys, aggregate_url, model_key, iterations):
    start_t = time.time()
    print("wait until all workers have registered. current worker: {}, total worker: {}"
          .format(worker_num, NUM_OF_WORKERS))
    while worker_num < int(NUM_OF_WORKERS):
        pass
    # start training
    training_urls = worker_ip_list
    for iter_num in range(iterations):
        # training enclaves
        threads = []
        for worker_id, worker_key in data_keys.items():
            t = threading.Thread(target=training_encalve_worker, args=(training_urls[worker_id], worker_id, iter_num))
            threads.append(t)
            t.start()
        mask_thread = threading.Thread(target=prepare_offline_masks_worker, args=[len(data_keys)])
        mask_thread.start()
        threads.append(mask_thread)
        for x in threads:
            x.join()  # make sure all threads finishes.

        # aggregate enclave
        response = requests.post(aggregate_url + "aggregate/" + str(prefix), json={'iter_num': iter_num})


def training_encalve_worker_mongo(url, worker_id, iter_num):
    response = requests.post("http://" + url + "/generate_gradient_mongo/" + str(worker_id), json={'iter_num': iter_num})
    return response.status_code

def start_worker_iteration(url, worker_id, iter_num):
    app.logger.info("sending iteration start siganl to worker_id {}, ip {}".format(worker_id, url))
    response = requests.get("http://" + url + "/start_iteration/" + str(iter_num))
    app.logger.info("finish sending start signal to worker_id {}, response {}".format(worker_id, response))
    

@app.route('/start_iteration/<iter_num>', methods=['POST', 'GET'])
def start_iteration(iter_num):
    iter_num = int(iter_num)
    global ITER_NUM
    
    if iter_num > ITER_NUM:
        app.logger.info("iter num exceed, stop. iter_num: {}".format(iter_num))
        return Response(status=200)
    
    training_urls = worker_ip_list
    start_t = time.time()
    threads = []
    app.logger.info("start to launch iteration {}".format(iter_num))
    for worker_id, worker_url in training_urls.items():
        t = threading.Thread(target=start_worker_iteration,
                             args=(worker_url, worker_id, iter_num))
        threads.append(t)
        t.start()
    for x in threads:
        x.join()

    elapsed_time = time.time() - start_t
    app.logger.info("finish to send launch iteration {} signal, finished in {:4f}s".format(iter_num, elapsed_time))
    return Response(status=200)

def training_thread_aggr_tree(prefix, data_keys, aggregate_url, model_key):
    time.sleep(5)

    start_t = time.time()
    app.logger.info("wait until all workers have registered. current worker: {}, total worker: {}"
          .format(worker_num, NUM_OF_WORKERS))
    while worker_num < int(NUM_OF_WORKERS):
        pass

    app.logger.info("start distribute ip_list, worker_num: {}".format(worker_num))
    worker_mgr_lock.acquire()
    app.logger.info("lock acquired")
    # distribute ip_list
    ip_list = []
    for i in range(worker_num):
        ip_list.append(worker_ip_list[i])
    ip_list_str = ",".join(ip_list)
    app.logger.info("ip_list: {}".format(ip_list_str))
    for i in range(worker_num):
        ip = worker_ip_list[i]
        response = requests.post("http://" + ip + "/receive_ip_list", json={"ip_list": ip_list_str})
        app.logger.info("distribute ip list to worker {}, ip {}: {}".format(i, ip, response))
        if response.status_code != 200:
            app.logger.error("distribute failed, stop")
            return
    # start training
    worker_mgr_lock.release()
    start_iteration(0)


def env_check():
    print("> NUM_OF_WORKERS: {}".format(NUM_OF_WORKERS))
    print("> AGGREGATOR_SVC_NAME: {}".format(AGGREGATOR_SVC_NAME))
    print("> STORAGE_SVC_NAME: {}".format(STORAGE_SVC_NAME))
    print("> SECUREML_NAMESPACE: {}".format(SECUREML_NAMESPACE))
    
    if NUM_OF_WORKERS is None or AGGREGATOR_SVC_NAME is None or SECUREML_NAMESPACE is None or STORAGE_SVC_NAME is None:
        print("> There is missing env var, exit.")
        return False
    return True


if __name__ == "__main__":
    if not env_check():
        exit(1)
#    prepare_offline_masks_mongo(NUM_OF_WORKERS, ITER_NUM)
    threading.Thread(target=training_thread_aggr_tree,
                     args=('this', DATA_KEYS, AGGREGATE_URL, MODEL_KEY)).start()
    app.run(debug=False, port=80, host='0.0.0.0', threaded=True)
