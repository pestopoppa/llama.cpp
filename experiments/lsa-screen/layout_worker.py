"""Fixture worker proving custom-op byte order and noncontiguous input strides."""
import struct
import sys


def exact(n):
    data = bytearray()
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return data


while True:
    header = sys.stdin.buffer.read(12)
    if not header:
        break
    nq, nk, layer = struct.unpack("=III", header)
    assert (nq, nk, layer) == (2, 32, 3)
    q = struct.unpack(f"={nq*24*256}f", exact(nq*24*256*4))
    k = struct.unpack(f"={nk*4*256}f", exact(nk*4*256*4))
    mask = exact(nq*nk*4)
    for i in range(nq):
        for h in range(24):
            for d in range(256):
                assert q[(i*24+h)*256+d] == i*100000+h*512+d
    for i in range(nk):
        for h in range(4):
            for d in range(256):
                assert k[(i*4+h)*256+d] == i*10000+h*512+d
    sys.stdout.buffer.write(struct.pack("=i", 0) + mask)
    sys.stdout.buffer.flush()
