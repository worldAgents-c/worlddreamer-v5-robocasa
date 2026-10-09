# Copyright (C) 2026 World Agents Corporation.
import io
import socket
import struct
import zipfile
import numpy as np

MAX_BYTES = 128 << 20


def _read(sock, count):
    result = bytearray()
    while len(result) < count:
        part = sock.recv(min(count - len(result), 1 << 20))
        if not part:
            raise EOFError('Connection closed during message')
        result.extend(part)
    return result


def send(sock, message):
    arrays = {key: np.asarray(value) for key, value in message.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Object arrays are not permitted')
    buffer = io.BytesIO(); np.savez(buffer, **arrays); payload = buffer.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError('Message exceeds the transport limit')
    sock.sendall(struct.pack('!Q', len(payload)) + payload)


def receive(sock):
    count, = struct.unpack('!Q', _read(sock, 8))
    if not 0 < count <= MAX_BYTES:
        raise ValueError('Invalid message length')
    buffer = io.BytesIO(_read(sock, count))
    with zipfile.ZipFile(buffer) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_BYTES or len(archive.infolist()) > 64:
            raise ValueError('Expanded message exceeds the transport limit')
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rpc(sock, **request):
    send(sock, request); response = receive(sock)
    if str(response.get('status')) != 'ok':
        raise RuntimeError(str(response.get('error', 'Invalid policy response')))
    return response


def connect(host, port):
    sock = socket.create_connection((host, port), timeout=30)
    sock.settimeout(1200)
    return sock
