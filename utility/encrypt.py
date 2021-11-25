import io
import time
import os
import uuid
import time

import numpy as np
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
from tempfile import TemporaryFile

from cryptography.fernet import Fernet

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
tmpdir = tempfile.mkdtemp()
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
MODEL_KEY = b''
encryptor_m = Fernet(MODEL_KEY)


def download_blob(container_name, blob_name, local_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    local_name = tmpdir + '/' + local_name
    with open(local_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return local_name


def upload_blob_file(container_name, local_file, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # local_file.seek(0)
    blob_client.upload_blob(local_file)


def upload_blob(container_name, data: bytes, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data)


def download_encypt_upload_blob(container_name, blob_name, local_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    local_name = tmpdir + '/' + local_name
    temp = blob_client.download_blob().readall()
    start_t = time.time()
    encrypted_model = encryptor_m.encrypt(temp)
    enc_t = time.time()
    decrpted_model = encryptor_m.decrypt(encrypted_model)
    dec_t = time.time()
    print("{}, {}".format(enc_t-start_t, dec_t-enc_t))
    # grad_file = TemporaryFile()
    # grad_file.write(encrypted_model)
    # upload_blob('model', encrypted_model, "alexnet_000")
    # upload_blob('model', decrpted_model, "alexnet_000_d")
    # upload_blob_file('model', grad_file, "alexnet_000____")
    return local_name


model_file = download_encypt_upload_blob('model', 'alexnet_init', 'alexnet_init')

