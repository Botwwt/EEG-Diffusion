"""Bounded loopback-only CPU service for an already verified idea gallery."""
from pathlib import Path
import argparse
import functools
import hashlib
import http.server
import json
import os
import signal
import threading
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    gallery = Path(args.gallery).resolve()
    receipt_path = Path(args.receipt).resolve()
    assert gallery.is_relative_to(DATA/'galleries')
    assert receipt_path.parent == ROOT/'audit' and not receipt_path.exists()
    proof = json.loads((gallery/'independent_materials_check_v1.json').read_text())
    assert proof['queries'] == 16600 and proof['pages'] == 415
    context_path = gallery/'comparison_context.json'
    with context_path.open('rb') as file:
        context_sha = hashlib.file_digest(file, 'sha256').hexdigest()
    assert context_sha == proof['comparison_context_sha256']
    os.nice(10)
    started = time.monotonic()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(gallery)))
    receipt = dict(status='RUNNING_LOOPBACK_ONLY_MAX600SECONDS', pid=os.getpid(),
        host='127.0.0.1', port=server.server_port, document_root=str(gallery),
        comparison_context_sha256=context_sha, GPU_used=False, publicly_exposed=False,
        started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    timer = threading.Timer(600, server.shutdown)
    timer.daemon = True
    timer.start()

    def stop(*args):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        timer.cancel()
        server.server_close()
        receipt.update(status='STOPPED_LOOPBACK_QA_SERVICE', seconds=time.monotonic()-started,
            ended_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
            file.write(json.dumps(dict(stage='idea_gallery_private_HTTP_QA_service',
                seconds=receipt['seconds'], GPU_used=False, actual_status=receipt['status'],
                receipt=str(receipt_path)))+'\n')


if __name__ == '__main__':
    main()
