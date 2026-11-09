"""Temporary loopback-only CPU server for existing full-gallery UI QA."""
from pathlib import Path
import functools
import http.server
import json
import os
import signal
import threading
import time

ROOT=Path(__file__).resolve().parent
GALLERY=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/galleries/V_existing_REG_patience20_complete_v1')


def main():
    os.nice(10)
    proof=json.loads((GALLERY/'independent_materials_check_v1.json').read_text())
    assert proof['queries']==16600 and proof['pages']==415
    path=ROOT/'audit/private_gallery_http_QA_service_v1.json'
    assert not path.exists(),'Preserve the prior service receipt'
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(GALLERY)))
    receipt=dict(status='RUNNING_LOOPBACK_ONLY_MAX600SECONDS',pid=os.getpid(),host='127.0.0.1',port=server.server_port,
        document_root=str(GALLERY),GPU_used=False,publicly_exposed=False,start_epoch=time.time())
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    timer=threading.Timer(600,server.shutdown);timer.daemon=True;timer.start()
    def stop(*args):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever(poll_interval=.2)
    finally:
        timer.cancel();server.server_close();receipt.update(status='STOPPED_LOOPBACK_QA_SERVICE',end_epoch=time.time())
        path.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
