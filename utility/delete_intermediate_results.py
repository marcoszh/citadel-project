import io
import time
import os
import uuid
import time

import numpy as np
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
from tempfile import TemporaryFile

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
tmpdir = tempfile.mkdtemp()
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# Create a unique name for the container
gradient_container_client = blob_service_client.get_container_client('gradients')
model_container_client = blob_service_client.get_container_client('model')


def delete_blob(container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.delete_blob()

blob_list = gradient_container_client.list_blobs(name_starts_with='masked_grads_')
for blob in blob_list:
    delete_blob('gradients', blob.name)

blob_list = model_container_client.list_blobs(name_starts_with='alexnet')
for blob in blob_list:
    if blob.name == 'alexnet_000':
        continue
    delete_blob('model', blob.name)