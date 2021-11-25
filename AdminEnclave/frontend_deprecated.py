import numpy as np
import zlib
import io
import time

import requests
from flask import Flask
from flask import request, Response
import threading

app = Flask(__name__)

MASK_SUM = None
MASK_SCALE = 100
SHAPES = [[3, 3, 3, 32], [32], [3, 3, 32, 32], [32], [3, 3, 32, 64], [64], [3, 3, 64, 64], [64], [2304, 512], [512], [512, 10], [10]]
MODEL_KEY = 
DATA_KEYS = 
DATA_BLOBS = {}
READY = False
TRAINING_ENCLAVE_ADDR = {0: "http://127.0.0.1:5003",}
AGGREGATE_ENCLAVE_ADDR = "http://127.0.0.1:5002"

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


@app.route('/req_mask/<worker_id>', methods=['POST'])
def request_mask(worker_id):
    global MASK_SUM
    start_t = time.time()
    if request.json['is_last']:
        mask = [0 - layer for layer in MASK_SUM]
    else:
        shapes = eval(request.json['shapes'])
        app.logger.debug('Processing mask request from {}'.format(worker_id))
        mask = []
        if not MASK_SUM:
            MASK_SUM = []
            for shape in shapes:
                temp = np.random.rand(*shape) * MASK_SCALE - 0.5 * MASK_SCALE
                mask.append(temp)
                MASK_SUM.append(np.copy(temp))
        else:
            for i, shape in enumerate(shapes):
                temp = np.random.rand(*shape) * MASK_SCALE - 0.5 * MASK_SCALE
                mask.append(temp)
                MASK_SUM[i] += temp

    bytestream = bytestream_nparr(mask)
    elapsed_time = time.time() - start_t
    app.logger.debug('Finished in {:4f}s'.format(elapsed_time))
    return Response(response=bytestream.getvalue(), status=200,
                    mimetype="application/octet_stream")


def training_encalve_worker(url, worker_id, iter_num, is_last):
    response = requests.post(url + "/generate_gradient/" + str(worker_id), json={'iter_num': iter_num, 'is_last': is_last})
    return response.status_code


def training_thread(prefix, data_keys, training_urls, aggregate_url, model_key, iterations):
    start_t = time.time()
    for iter_num in range(iterations):
        # training enclaves
        threads = []
        for worker_id, worker_key in data_keys.items():
            is_last = (worker_id == (len(data_keys) - 1))
            t = threading.Thread(target=training_encalve_worker, args=(training_urls[worker_id], worker_id, iter_num, is_last))
            threads.append(t)
            t.start()
        for x in threads:
            x.join()  # make sure all threads finishes.

        # aggregate enclave
        response = requests.post(aggregate_url + "/aggregate/" + str(prefix), json={'iter_num': iter_num})


if __name__ == "__main__":
    threading.Thread(target=training_thread,
                     args=('this', DATA_KEYS, TRAINING_ENCLAVE_ADDR, AGGREGATE_ENCLAVE_ADDR, MODEL_KEY, 5)).start()
    app.run(debug=False, port=5001, host='0.0.0.0')
