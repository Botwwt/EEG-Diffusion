"""Small persistent SSH RPC; credentials arrive over stdin and never enter files."""
import json
import getpass
import argparse
import socket
import threading
from pathlib import Path
import sys
import time

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='CLOUD_LOGIN_HOST')
parser.add_argument('--port', type=int, default=54067)
parser.add_argument('--http-proxy', help='Local HTTP CONNECT proxy, e.g. 127.0.0.1:2080')
parser.add_argument('--public-relay-port', type=int, help='Optional remote loopback relay, HTTPS Hugging Face only')
args = parser.parse_args()
assert args.host.endswith('.seetacloud.com') and 1 <= args.port <= 65535
credentials = ({'password': getpass.getpass('SSH password: ')} if sys.stdin.isatty()
               else json.loads(sys.stdin.readline()))
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
tunnel = None
if args.http_proxy:
    proxy_host, proxy_port = args.http_proxy.rsplit(':', 1)
    assert proxy_host in ('127.0.0.1', 'localhost'), 'Use the configured local proxy'
    tunnel = socket.create_connection((proxy_host, int(proxy_port)), timeout=15)
    tunnel.settimeout(25)
    destination = f'{args.host}:{args.port}'
    tunnel.sendall(('CONNECT ' + destination + ' HTTP/1.1\r\nHost: ' + destination + '\r\n\r\n').encode('ascii'))
    response = b''
    while b'\r\n\r\n' not in response:
        byte = tunnel.recv(1)
        if not byte:
            raise ConnectionError('Proxy closed before CONNECT response')
        response += byte
        if len(response) > 16384:
            raise ConnectionError('Oversized proxy response')
    if response.split(b'\r\n', 1)[0].split()[1] != b'200':
        raise ConnectionError('Proxy CONNECT failed: ' + response.split(b'\r\n', 1)[0].decode('ascii', 'replace'))
client.connect(args.host, port=args.port, username='root',
               password=credentials.pop('password'), timeout=25, banner_timeout=25,
               sock=tunnel)
client.get_transport().set_keepalive(15)
if args.public_relay_port:
    assert args.http_proxy and 1024 <= args.public_relay_port < 65536

    def relay(channel, origin, server):
        def worker():
            peer = None
            try:
                channel.settimeout(60)
                header = b''
                while b'\r\n\r\n' not in header:
                    byte = channel.recv(1)
                    if not byte:
                        return
                    header += byte
                    assert len(header) <= 16384
                method, destination, version = header.split(b'\r\n', 1)[0].decode('ascii').split()
                host, port = destination.rsplit(':', 1)
                assert method == 'CONNECT' and port == '443'
                assert any(host == domain or host.endswith('.' + domain)
                           for domain in ('huggingface.co', 'hf.co'))
                peer = socket.create_connection((proxy_host, int(proxy_port)), timeout=15)
                peer.settimeout(60)
                peer.sendall(('CONNECT ' + destination + ' HTTP/1.1\r\nHost: ' + destination + '\r\n\r\n').encode('ascii'))
                reply = b''
                while b'\r\n\r\n' not in reply:
                    byte = peer.recv(1)
                    assert byte
                    reply += byte
                    assert len(reply) <= 16384
                assert reply.split(b'\r\n', 1)[0].split()[1] == b'200'
                channel.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')

                def pump(source, target):
                    try:
                        while True:
                            block = source.recv(65536)
                            if not block:
                                break
                            target.sendall(block)
                    except (OSError, EOFError):
                        pass
                    finally:
                        channel.close()
                        peer.close()

                threading.Thread(target=pump, args=(channel, peer), daemon=True).start()
                pump(peer, channel)
            except Exception:
                channel.close()
                if peer is not None:
                    peer.close()
        threading.Thread(target=worker, daemon=True).start()

    client.get_transport().request_port_forward('127.0.0.1', args.public_relay_port, handler=relay)
    print(json.dumps({'status': 'PUBLIC_HF_ONLY_LOOPBACK_RELAY_READY', 'remote_port': args.public_relay_port}), flush=True)
print(json.dumps({'status': 'CONNECTED'}), flush=True)
log = Path(__file__).parent / 'audit/ssh_rpc_commands.jsonl'
for line in sys.stdin:
    request = json.loads(line)
    try:
        op = request['op']
        if op == 'close':
            break
        if op == 'run':
            si, so, se = client.exec_command(request['cmd'], timeout=request.get('timeout', 60))
            si.close()
            stdout = so.read().decode('utf-8', errors='replace')
            stderr = se.read().decode('utf-8', errors='replace')
            result = dict(code=so.channel.recv_exit_status(), stdout=stdout, stderr=stderr)
        elif op in ('put', 'get'):
            with client.open_sftp() as sftp:
                if op == 'put':
                    sftp.put(request['local'], request['remote'])
                else:
                    sftp.get(request['remote'], request['local'])
            result = dict(status='TRANSFERRED', op=op, remote=request['remote'])
        else:
            raise ValueError(op)
        record = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      request=request, result=result)
        with log.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps(dict(error=type(exc).__name__, message=str(exc))), flush=True)
client.close()
