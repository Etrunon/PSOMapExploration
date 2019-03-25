import zlib
import base64

f = open('./bio/decoded_exstr.txt', 'rb')
decompressed_data = zlib.decompress(f)
