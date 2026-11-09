"""Test completed idea-gallery UI over an encrypted ephemeral SSH forward.

Only small screenshots and receipts are saved locally. Model weights, datasets,
checkpoints and the full gallery remain on the server. No human ratings.
"""
from pathlib import Path
import argparse
import getpass
import hashlib
import json
import select
import socketserver
import sys
import threading
import time

sys.path.insert(0, r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote-port', type=int, required=True)
    parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--chrome', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-source', action='append', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT/'audit') and output != ROOT/'audit'
    assert not output.exists() and Path(args.chrome).is_file()
    assert 1 <= args.remote_port <= 65535 and len(args.context_sha256) == 64
    output.mkdir()
    started = time.monotonic()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('CLOUD_LOGIN_HOST', port=54067, username='root',
        password=getpass.getpass('SSH password: '), timeout=25, banner_timeout=25,
        look_for_keys=False, allow_agent=False)
    transport = client.get_transport()
    transport.set_keepalive(15)

    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            channel = transport.open_channel('direct-tcpip',
                ('127.0.0.1', args.remote_port), self.request.getpeername())
            try:
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [], 30)
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            finally:
                channel.close()

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        request_queue_size = 32

    server = Server(('127.0.0.1', 0), Forward)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_address[1]}'
    report = dict(status='RUNNING_BROWSER_QA', GPU_used=False, public_service=False,
        encrypted_SSH_forward=True, bulk_downloads=False, human_ratings_performed=False,
        scientific_goal_complete=False, checks=[], screenshots=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        browser_fallback='Installed Playwright and Chrome headless; native browser runtime previously failed kernel asset initialization')
    browser = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.chrome, headless=True,
                args=['--disable-gpu', '--disk-cache-size=1', '--media-cache-size=1',
                    '--disable-application-cache', '--disable-background-networking'])
            context = browser.new_context(viewport=dict(width=1600, height=1000),
                accept_downloads=False, service_workers='block')
            external, errors, failed = [], [], []

            def route(request):
                if request.request.url.startswith(origin+'/'):
                    request.continue_()
                else:
                    external.append(request.request.url)
                    request.abort()

            context.route('**/*', route)
            response = context.request.get(origin+'/comparison_context.json')
            assert response.status == 200
            raw = response.body()
            assert hashlib.sha256(raw).hexdigest() == args.context_sha256
            source = json.loads(raw)
            assert source['complete_sources'] == args.expected_source and len(args.expected_source) >= 2
            report.update(comparison_context_sha256=args.context_sha256,
                complete_sources=source['complete_sources'],
                missing_required_sources=source['missing_required_sources'])
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('requestfailed', lambda request: failed.append(dict(url=request.url, error=request.failure)))
            response = page.goto(origin+'/index.html', wait_until='networkidle', timeout=45000)
            assert response.status == 200

            def count():
                return page.locator('#count').inner_text()

            def first():
                return page.locator('#images tr').nth(1).locator('code').inner_text()

            def image_ready():
                page.wait_for_function('Array.from(document.querySelectorAll("img")).filter(i=>i.getBoundingClientRect().top<1000).every(i=>i.complete&&i.naturalWidth>0)')

            def screenshot(name):
                path = output/name
                page.screenshot(path=str(path), type='jpeg', quality=70)
                assert path.stat().st_size < 1024**2
                report['screenshots'].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

            assert count().startswith('16600 / 16600') and '415' in count()
            assert page.locator('#images tr').count() == 41
            assert page.locator('#images th').all_text_contents()[1:] == source['complete_sources']
            subjects = page.locator('#subject option').evaluate_all('(nodes)=>nodes.map(n=>n.value).filter(Boolean)')
            concepts = page.locator('#concept option').evaluate_all('(nodes)=>nodes.map(n=>n.value).filter(Boolean)')
            assert len(subjects) == 10 and len(concepts) == 166
            body = page.locator('body').inner_text()
            assert all(name in body for name in source['missing_required_sources'])
            image_ready()
            screenshot('unblinded_first_stable_page.jpg')
            report['checks'].append(dict(name='complete_population_and_missing_controls_notice', queries=16600, subjects=10, concepts=166, pages=415))
            first_id = first()
            page.get_by_role('button', name='下一页', exact=True).click()
            assert '第 2 / 415' in count() and first() != first_id
            page.get_by_role('button', name='上一页', exact=True).click()
            assert first() == first_id
            page.get_by_role('button', name='上一页', exact=True).click()
            assert '第 1 / 415' in count()
            page.locator('#subject').select_option(subjects[0])
            assert count().startswith('1660 / 16600')
            page.locator('#concept').select_option(concepts[0])
            assert count().startswith('10 / 16600')
            query = first()
            page.locator('#search').fill(query)
            assert count().startswith('1 / 16600') and page.locator('#images tr').count() == 2
            case = page.evaluate('(q)=>window.GALLERY.rows.find(r=>r.id===q)', query)
            assert case['id'] == first_id and len(case['cells']) == len(args.expected_source)
            (output/'first_stable_query_case.json').write_text(json.dumps(case, indent=2)+'\n', encoding='utf-8')
            image_ready()
            screenshot('first_stable_query_pair.jpg')
            for relative in [case['targetFull'], *[cell['full'] for cell in case['cells']]]:
                response = context.request.get(origin+'/'+relative)
                assert response.status == 200 and response.headers['content-type'].startswith('image/')
            report['checks'].append(dict(name='pagination_filters_fixed_query_and_full_image_links', query=query, passed=True))
            page.locator('#search').fill('')
            page.locator('#concept').select_option('')
            for _ in range(41):
                page.get_by_role('button', name='下一页', exact=True).click()
            assert '第 42 / 42' in count() and page.locator('#images tr').count() == 21
            page.get_by_role('button', name='下一页', exact=True).click()
            assert '第 42 / 42' in count()
            report['checks'].append(dict(name='subject_final_page_boundary', passed=True))
            response = page.goto(origin+'/pages/index.html', wait_until='networkidle')
            assert response.status == 200 and page.locator('li a').count() == 415
            page.locator('li a').first.click()
            assert page.locator('table tr').count() == 41
            image_ready()
            report['checks'].append(dict(name='all_static_page_links_and_first_static_render', page_links=415, passed=True))
            response = page.goto(origin+'/case_indices.html', wait_until='networkidle')
            assert response.status == 200 and '待完成配对比较' in page.locator('body').inner_text()
            paired_links = page.locator('li a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))')
            assert len(paired_links) >= 2 and any('largest_gain' in p for p in paired_links) and any('largest_decline' in p for p in paired_links)
            for relative in paired_links:
                assert relative.endswith('_all_cases.csv') and '/' not in relative
                assert context.request.head(origin+'/'+relative).status == 200
            report['checks'].append(dict(name='complete_paired_gain_and_decline_indices', links=len(paired_links), passed=True))
            response = page.goto(origin+'/blinded/index.html', wait_until='networkidle', timeout=45000)
            assert response.status == 200 and count().startswith('16600 / 16600')
            assert page.locator('#images th').all_text_contents() == ['目标 / ID', *[chr(65+i) for i in range(len(args.expected_source))]]
            text = page.locator('body').inner_text()
            assert all(value not in text for value in ['Conditional', 'RealPaired', 'DINO2AFC', 'SEED', 'ObjectF1'])
            page.locator('#subject').select_option(subjects[0])
            assert count().startswith('1660 / 16600')
            report['checks'].append(dict(name='complete_blinded_population_hidden_methods_and_scores', passed=True))
            unexpected = [item for item in failed if 'ERR_ABORTED' not in str(item['error'])]
            assert not errors and not external and not unexpected, (errors, external, unexpected)
            report.update(status='PASSED_ACTUAL_BROWSER_UI_CHECKS', browser_version=browser.version,
                JS_errors=errors, external_request_attempts=external, failed_requests=unexpected,
                expected_navigation_or_filter_cancelled_requests=len(failed)-len(unexpected),
                standalone_file_protocol_opened=False, limits='UI and links verified; no human quality scores, paired gain or scientific success is inferred.')
            browser.close()
            browser = None
    except Exception as error:
        report.update(status='FAILED_BROWSER_QA_PRESERVE_EVIDENCE', error=repr(error))
        raise
    finally:
        if browser is not None:
            browser.close()
        server.shutdown()
        server.server_close()
        client.close()
        report.update(seconds=time.monotonic()-started)
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
