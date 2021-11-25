import io
import numpy as np
import zlib
import requests


# ## CONFIG

SERVER_HOST= "localhost"
SERVER_PORT = 5000


# ## HELPERS

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


# ## MAIN CLIENT ROUTINE

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+'/req_mask/idsfds'
while True:
    resp = requests.get(url)
    #
    print("\nresponse:")
    data = uncompress_nparr(resp.content)
    print(data)