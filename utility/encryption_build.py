from cffi import FFI


ffibuilder = FFI()

ffibuilder.cdef(r"""
int encrypt_ecb(unsigned char * input, unsigned char * output,
                unsigned char * key, int len);
int decrypt_ecb(unsigned char * input, unsigned char * output,
                unsigned char * key, int len);
""")


ffibuilder.set_source("_encryption",
r"""
#include <openssl/evp.h>
int encrypt_ecb(unsigned char * input, unsigned char * output,
                unsigned char * key, int len)
{
  int outlen, finallen;
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  EVP_CIPHER_CTX_init(ctx);
  EVP_EncryptInit(ctx, EVP_aes_128_ecb(), key, 0);
  // EVP_CIPHER_CTX_set_padding(ctx, 0);
  if(!EVP_EncryptUpdate(ctx, output, &outlen, input, len)) return 0;
  if(!EVP_EncryptFinal(ctx, output + outlen, &finallen)) return 0;
  EVP_CIPHER_CTX_free(ctx);
  return outlen + finallen;
}

int decrypt_ecb(unsigned char * input, unsigned char * output,
                unsigned char * key, int len)
{
  int outlen, finallen;
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  EVP_CIPHER_CTX_init(ctx);
  EVP_DecryptInit(ctx, EVP_aes_128_ecb(), key, 0);
  // EVP_CIPHER_CTX_set_padding(ctx, 0);
  if(!EVP_DecryptUpdate(ctx, output, &outlen, input, len)) return 0;
  if(!EVP_DecryptFinal(ctx, output + outlen, &finallen)) return 0;
  EVP_CIPHER_CTX_free(ctx);
  return outlen + finallen;
}
            
""", libraries=["crypto"], extra_compile_args=['-Wno-deprecated-declarations'])


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)