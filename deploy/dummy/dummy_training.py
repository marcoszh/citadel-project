import os
import atexit
import requests

from flask import Flask

app = Flask(__name__)

# For k8s deployment
POD_IP = os.getenv("KUBE_POD_IP")
FRONTEND_SVC_NAME = os.getenv("FRONTEND_SVC_NAME")
AGGREGATOR_SVC_NAME = os.getenv("AGGREGATOR_SVC_NAME")
SECUREML_NAMESPACE = os.getenv("SECUREML_NAMESPACE")

REG_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/register_worker/'
UNREG_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/unregister_worker/'
MASK_URL = f'http://{FRONTEND_SVC_NAME}.{SECUREML_NAMESPACE}/req_mask/'
AGGREGATE_URL = f'http://{AGGREGATOR_SVC_NAME}.{SECUREML_NAMESPACE}/upload_gradient/'

def env_check():
    print("> POD_IP: {}".format(POD_IP))
    print("> FRONTEND_SVC_NAME: {}".format(FRONTEND_SVC_NAME))
    print("> AGGREGATOR_SVC_NAME: {}".format(AGGREGATOR_SVC_NAME))
    print("> SECUREML_NAMESPACE: {}".format(SECUREML_NAMESPACE))
    if POD_IP is None or FRONTEND_SVC_NAME is None or AGGREGATOR_SVC_NAME is None or SECUREML_NAMESPACE is None:
        print("> There is missing env var, exit.")
        return False
    return True


def register_this_worker():
    print('> register this worker {}'.format(POD_IP))
    r = requests.get(REG_URL + POD_IP).json()
    print('> register result: {}'.format(r))
    if "id" in r:
        worker_id = r["id"]
    if worker_id != -1:
        print('> register succeeded, id: {}'.format(r["id"]))


def unregister_this_worker():
    print('> unregister this worker, ip: {}'.format(POD_IP))
    r = requests.get(UNREG_URL + POD_IP).json()
    print('> unregister result: {}'.format(r))


if __name__ == "__main__":
    if not env_check():
        exit(1)
    register_this_worker()
    atexit.register(unregister_this_worker)
    app.run(debug=False, port=80, host='0.0.0.0')
