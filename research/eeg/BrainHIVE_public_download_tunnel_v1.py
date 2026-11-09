"""Temporary SSH loopback-only CONNECT relay for public model downloads.

Only Hugging Face domains on HTTPS are allowed. No local artifact payloads,
EEG, credentials, or model files are written by this relay.
"""
import getpass
import select
import socket
import sys
import threading
import time
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
            first = header.split(b'\r\n', 1)[0].decode('ascii')
            method, address, version = first.split()
            host, port_text = address.rsplit(':', 1)
            assert method == 'CONNECT' and int(port_text) == 443
            assert any(host == domain or host.endswith('.' + domain) for domain in ('huggingface.co', 'hf.co'))
            remote = socket.create_connection((host, 443), timeout=20)
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
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass('SSH password: ')
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root', password=password,
                   timeout=25, banner_timeout=25)
    password = None
    transport = client.get_transport()
    transport.set_keepalive(15)
    port = transport.request_port_forward('127.0.0.1', 33343, handler=relay)
    print('PUBLIC_ONLY_RELAY_READY_REMOTE_LOOPBACK', port, flush=True)
    try:
        for _ in range(240):  # Finite two-hour lease, closes early when the current transfer completes.
            time.sleep(30)
            if not transport.is_active():
                break
            _, stdout, _ = client.exec_command(
                "test -f /root/brainhive_reproduction_20261020_v1/assets/transfer_complete.json", timeout=10)
            if stdout.channel.recv_exit_status() == 0:
                break
    finally:
        if transport.is_active():
            transport.cancel_port_forward('127.0.0.1', port)
        client.close()
    print('PUBLIC_ONLY_RELAY_CLOSED', flush=True)


if __name__ == '__main__':
    main()
