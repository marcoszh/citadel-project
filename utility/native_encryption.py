from ._encryption import ffi, lib


def encrypt_ecb_cffi(data, key):
    datalen = len(data)
    out = ffi.new("char[%s]" % (datalen))
    num = lib.encrypt_ecb(data, out, key, datalen)
    return ffi.unpack(out, num)


def decrypt_ecb_cffi(data, key):
    datalen = len(data)
    out = ffi.new("char[%s]" % (datalen))
    num = lib.decrypt_ecb(data, out, key, datalen)
    return ffi.unpack(out, num)


class NativeEncryptor:
    def __init__(self, key=b"needstobesixteen"):
        self.key = key

    def encrypt(self, data):
        return encrypt_ecb_cffi(data, self.key)

    def decrypt(self, data):
        return decrypt_ecb_cffi(data, self.key)
