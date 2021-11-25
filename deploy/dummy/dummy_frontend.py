import os
import time

import requests
from flask import Flask
from flask import request, Response, jsonify
import threading

app = Flask(__name__)

# For k8s deploy
NUM_OF_WORKERS = os.getenv("NUM_OF_WORKERS")
AGGREGATOR_SVC_NAME = os.getenv("AGGREGATOR_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")

AGGREGATE_URL = f'{AGGREGATOR_SVC_NAME}.{SECUREML_NAMESPACE}.svc.cluster.local/upload_gradient/'

# For worker management
worker_mgr_lock = threading.Lock()
worker_ip_to_index = dict()
worker_ip_list = dict()
worker_num = 0

start = False

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
    worker_num += 1
    app.logger.debug('Worker {} register succeeded, id: {}, current worker: {}'.format(worker_ip, idx, worker_num))
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
    worker_num -= 1

    worker_mgr_lock.release()
    app.logger.debug('Worker {} unregistered'.format(worker_ip))
    return jsonify({
        "msg": "unregistered"
    })

@app.route('/ping', methods=['GET'])
def ping():
    if start:
        return "OK"
    return "not OK"

def training_thread():
    global worker_num, start
    start_t = time.time()
    print("wait until all workers have registered. current worker: {}, total worker: {}"
                    .format(worker_num, NUM_OF_WORKERS))
    while worker_num < int(NUM_OF_WORKERS):
        pass
    start = True
    print("start training ...")
    print("ip list: {}".format(worker_ip_list))
    exit(0)


if __name__ == "__main__":
    threading.Thread(target=training_thread,
                     args=[]).start()
    app.run(debug=True, port=80, host='0.0.0.0')
