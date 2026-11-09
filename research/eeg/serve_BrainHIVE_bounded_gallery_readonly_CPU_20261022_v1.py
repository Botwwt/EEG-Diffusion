"""Finite loopback-only UI QA server for the completed old gallery.

Only the exact saved index outputs, old pages and registered image URLs are
served. No directory listing, arbitrary filesystem route, write method or GPU.
"""
from pathlib import Path
import functools
import hashlib
import http.server
import json
import os
import posixpath
import signal
import time
from urllib.parse import unquote, urlsplit

import brainhive_functional_training_contract_v2 as C

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/brainhive_reproduction_20261020_v1/gallery')
INDEX = BASE / 'bounded_component_V_index_20261022_v1'
FOLDER = ROOT / 'audit/gallery_browser_QA_20261022_v1'
PORT, SECONDS, MIN_MEMORY = 38691, 1800, 12 << 30


def write(path, value):
    encoded = (json.dumps(value, indent=2) + '\n').encode()
    assert len(encoded) < 1 << 20
    tmp = path.with_suffix('.pending')
    with tmp.open('wb') as stream:
        stream.write(encoded)
    tmp.replace(path)


def main():
    assert os.name == 'posix'
    assert C.resource_headroom()['effective_headroom_bytes'] >= MIN_MEMORY
    evidence = ROOT / 'audit/gallery_index_actual_20261022_v1/delivery_verification_v3.json'
    assert C.sha(evidence) == 'e90da14961e491356b7ecfc007d5a29b83f09b0cda6f1262324c904fd7478fcb'
    assert json.loads(evidence.read_text())['status'] == 'FULL_OLD_THREE_ARM_GALLERY_DELIVERY_VERIFIED'
    completion = json.loads((INDEX / 'completion.json').read_text())
    assert completion['queries'] == 16600 and completion['generated_images'] == 49800
    for name, expected in completion['file_sha256'].items():
        assert C.sha(INDEX / name) == expected
    raw = (INDEX / 'data.js').read_text()
    assert raw.startswith('window.GALLERY=') and raw.endswith(';\n')
    data = json.loads(raw[len('window.GALLERY='):-2])
    assert len(data['rows']) == 16600
    allowed = {'/' + INDEX.name + '/' + name for name in completion['file_sha256']}
    allowed.add('/' + INDEX.name + '/completion.json')
    for i in range(166):
        allowed.add('/bounded_component_V_v1/page_' + str(i).zfill(3) + '.html')
    for row in data['rows']:
        urls = [row['target'], row['targetFull'], row['pageLocator']]
        urls += [cell[key] for cell in row['cells'] for key in ('full', 'thumb')]
        for url in urls:
            parsed = urlsplit(url)
            assert not parsed.scheme and not parsed.netloc and not parsed.query
            route = posixpath.normpath('/' + INDEX.name + '/' + unquote(parsed.path))
            assert route.startswith('/bounded_component_V_v1/')
            assert (BASE / route.lstrip('/')).is_file()
            allowed.add(route)
    del data, raw
    FOLDER.mkdir(exist_ok=True)
    receipt = FOLDER / 'server.json'
    assert not receipt.exists(), 'Keep prior UI QA evidence; no automatic retry'
    stats = dict(status='LOOPBACK_READONLY_GALLERY_UI_QA_SERVER_RUNNING',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
        source_sha256=C.sha(__file__), host='127.0.0.1', port=PORT, maximum_seconds=SECONDS,
        allowlisted_routes=len(allowed), requests=0, served_bytes=0, denied=0,
        image_generation=False, new_scores_computed=False, GPU_used=False,
        original_files_modified=False, public_listener=False)
    write(receipt, stats)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def list_directory(self, _path):
            self.send_error(403)
            return None

        def send_head(self):
            stats['requests'] += 1
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            if route == '/favicon.ico':
                self.send_response(204); self.end_headers(); return None
            if parsed.query or route not in allowed or '..' in route.split('/'):
                stats['denied'] += 1; self.send_error(403); return None
            if C.resource_headroom()['effective_headroom_bytes'] < MIN_MEMORY:
                stats['denied'] += 1; self.send_error(503, 'CPU headroom guard'); return None
            return super().send_head()

        def copyfile(self, source, output):
            while True:
                block = source.read(64 << 10)
                if not block:
                    return
                output.write(block); stats['served_bytes'] += len(block)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    server = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), functools.partial(Handler, directory=str(BASE)))
    server.timeout = .5
    started = time.monotonic()
    def stop(_signum, _frame):
        raise KeyboardInterrupt('Finite UI QA server stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(SECONDS)
    try:
        while time.monotonic() - started < SECONDS:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        signal.alarm(0); server.server_close()
        stats.update(status='LOOPBACK_READONLY_GALLERY_UI_QA_SERVER_STOPPED', seconds=time.monotonic()-started,
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            memory_after=C.resource_headroom(), browser_verdict_recorded_separately=True)
        write(receipt, stats)


if __name__ == '__main__':
    main()
