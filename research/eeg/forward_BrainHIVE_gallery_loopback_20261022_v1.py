"""Temporary loopback-only SSH forwarding for read-only gallery browser QA.

Credentials are entered through getpass and are never logged or persisted.
The only destination is the remote private gallery HTTP listener.
"""
import getpass
import json
import select
import socket
import socketserver
import sys
import threading
import time

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko

HOST, SSH_PORT = 'CLOUD_LOGIN_HOST', 23526
LOCAL_PORT, REMOTE_PORT, SECONDS = 38691, 38691, 1800
sock = socket.create_connection(('127.0.0.1', 2080), timeout=15)
sock.sendall(('CONNECT ' + HOST + ':' + str(SSH_PORT) + ' HTTP/1.1\r\nHost: ' + HOST + '\r\n\r\n').encode())
reply = b''
while b'\r\n\r\n' not in reply:
    reply += sock.recv(1)
    assert len(reply) < 16384
assert reply.split(b'\r\n', 1)[0].split()[1] == b'200'
client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=SSH_PORT, username='root', password=getpass.getpass('SSH password: '), sock=sock,
    timeout=25, banner_timeout=25)
client.get_transport().set_keepalive(15)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        channel = client.get_transport().open_channel('direct-tcpip', ('127.0.0.1', REMOTE_PORT), self.client_address,
            timeout=15)
        try:
            while True:
                ready, _, _ = select.select([self.request, channel], [], [], 30)
                if not ready:
                    break
                for source in ready:
                    block = source.recv(64 << 10)
                    if not block:
                        return
                    (channel if source is self.request else self.request).sendall(block)
        finally:
            channel.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


with Server(('127.0.0.1', LOCAL_PORT), Handler) as server:
    server.timeout = .5
    print(json.dumps(dict(status='PRIVATE_GALLERY_LOOPBACK_SSH_FORWARD_READY', local_host='127.0.0.1',
        local_port=LOCAL_PORT, remote_host='127.0.0.1', remote_port=REMOTE_PORT, maximum_seconds=SECONDS)), flush=True)
    deadline = time.monotonic() + SECONDS
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    except KeyboardInterrupt:
        pass
client.close()
