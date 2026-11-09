"""Finite private SSH relay for public HF downloads through the current local proxy.

No model payloads are saved locally. Only HTTPS Hugging Face destinations are
accepted. The original direct-connect relay remains preserved as version 1.
"""
import getpass
import select
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko


def relay(channel, origin, server):
    def worker():
        remote = None
        try:
            header = b''
            while b'\r\n\r\n' not in header:
                part = channel.recv(4096)
                if not part:
                    return
                header += part
                assert len(header) <= 16384
            method, address, version = header.split(b'\r\n', 1)[0].decode('ascii').split()
            host, port_text = address.rsplit(':', 1)
            assert method == 'CONNECT' and int(port_text) == 443
            assert any(host == d or host.endswith('.' + d) for d in ('huggingface.co', 'hf.co'))
            proxy_url = urllib.request.getproxies().get('https')
            assert proxy_url, 'Current local HTTPS proxy is required'
            proxy = urllib.parse.urlsplit(proxy_url)
            assert proxy.scheme == 'http' and proxy.hostname in ('127.0.0.1', 'localhost')
            assert proxy.username is None and proxy.password is None
            remote = socket.create_connection((proxy.hostname, proxy.port or 80), timeout=30)
            remote.sendall(('CONNECT ' + address + ' HTTP/1.1\r\nHost: ' + address + '\r\n\r\n').encode('ascii'))
            reply = b''
            while b'\r\n\r\n' not in reply:
                part = remote.recv(4096)
                assert part, 'Proxy closed before CONNECT response'
                reply += part
                assert len(reply) <= 16384
            proxy_header, extra = reply.split(b'\r\n\r\n', 1)
            assert proxy_header.split(b'\r\n', 1)[0].split()[1] == b'200'
            assert not extra, 'Unexpected bytes before TLS client handshake'
            remote.settimeout(None)
            channel.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
            while True:
                readable, _, _ = select.select([channel, remote], [], [], 30)
                for stream in readable:
                    data = stream.recv(262144)
                    if not data:
                        return
                    (remote if stream is channel else channel).sendall(data)
        except Exception as error:
            print('PUBLIC_RELAY_CONNECTION_ENDED', type(error).__name__, flush=True)
        finally:
            channel.close()
            if remote is not None:
                remote.close()
    threading.Thread(target=worker, daemon=True).start()


def main():
    proxy = urllib.parse.urlsplit(urllib.request.getproxies().get('https', ''))
    assert proxy.scheme == 'http' and proxy.hostname in ('127.0.0.1', 'localhost')
    print('CURRENT_LOCAL_HTTP_PROXY_PORT', proxy.port or 80, flush=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass('SSH password: ')
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root', password=password,
                   timeout=25, banner_timeout=25)
    password = None
    transport = client.get_transport()
    transport.set_keepalive(15)
    port = transport.request_port_forward('127.0.0.1', 33343, handler=relay)
    print('PUBLIC_ONLY_PROXY_RELAY_READY_REMOTE_LOOPBACK', port, flush=True)
    try:
        for tick in range(240):
            time.sleep(30)
            if not transport.is_active():
                break
            if tick % 10 == 9:
                # A busy reverse tunnel can delay a new SSH control channel.
                # A completion-check timeout must never close active transfers.
                try:
                    _, stdout, _ = client.exec_command(
                        'test -f /root/brainhive_reproduction_20261020_v1/assets/transfer_complete.json', timeout=20)
                    if stdout.channel.recv_exit_status() == 0:
                        break
                except (paramiko.SSHException, socket.timeout):
                    print('COMPLETION_CHECK_DEFERRED_RELAY_REMAINS_ACTIVE', flush=True)
    finally:
        if transport.is_active():
            transport.cancel_port_forward('127.0.0.1', port)
        client.close()
    print('PUBLIC_ONLY_PROXY_RELAY_CLOSED', flush=True)


if __name__ == '__main__':
    main()
