"""Actual headless UI checks over an ephemeral encrypted SSH forward.

Uses installed Chrome/Playwright, no model/data/gallery bulk downloads or public
service. Credentials stay in memory. Only small screenshots/reports are saved.
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

sys.path.insert(0,r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import paramiko
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--remote-port',type=int,required=True)
    parser.add_argument('--chrome',required=True);args=parser.parse_args()
    assert Path(args.chrome).is_file() and 1<=args.remote_port<=65535
    output=ROOT/'audit/gallery_browser_QA_v1';assert not output.exists();output.mkdir()
    started=time.monotonic();client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('CLOUD_LOGIN_HOST',port=54067,username='root',password=getpass.getpass('SSH password: '),
        timeout=25,banner_timeout=25,look_for_keys=False,allow_agent=False)
    transport=client.get_transport();transport.set_keepalive(15)
    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            channel=transport.open_channel('direct-tcpip',('127.0.0.1',args.remote_port),self.request.getpeername())
            try:
                while True:
                    readable,_,_=select.select([self.request,channel],[],[],30)
                    for source in readable:
                        data=source.recv(65536)
                        if not data:return
                        (channel if source is self.request else self.request).sendall(data)
            finally:channel.close()
    class Server(socketserver.ThreadingTCPServer):
        daemon_threads=True
        request_queue_size=32
    server=Server(('127.0.0.1',0),Forward)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    origin=f'http://127.0.0.1:{server.server_address[1]}'
    report=dict(status='RUNNING_BROWSER_QA',historical_patience20_reference_only=True,GPU_used=False,
        native_browser_tool_failure='failed to write kernel assets: system path not found; retried after reset',
        browser_fallback='Installed Playwright Python and installed Chrome headless',
        encrypted_SSH_forward=True,public_service=False,bulk_downloads=False,checks=[],screenshots=[])
    browser=None
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path=args.chrome,headless=True,
                args=['--disable-gpu','--disk-cache-size=1','--media-cache-size=1','--disable-application-cache','--disable-background-networking'])
            context=browser.new_context(viewport=dict(width=1600,height=1000),accept_downloads=False,service_workers='block')
            external=[];errors=[];failed=[]
            def route(request):
                if request.request.url.startswith(origin+'/'):request.continue_()
                else:external.append(request.request.url);request.abort()
            context.route('**/*',route)
            page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
            page.on('requestfailed',lambda request:failed.append(dict(url=request.url,error=request.failure)))
            response=page.goto(origin+'/index.html',wait_until='networkidle',timeout=45000);assert response.status==200
            assert page.locator('#count').count()==page.locator('#subject').count()==page.locator('#concept').count()==1
            def count():return page.locator('#count').inner_text()
            def first():return page.locator('#images tr').nth(1).locator('code').inner_text()
            initial=count();assert initial.startswith('16600 / 16600') and '415' in initial
            assert page.locator('#images tr').count()==41
            subjects=page.locator('#subject option').evaluate_all('(nodes)=>nodes.map(n=>n.value).filter(Boolean)')
            concepts=page.locator('#concept option').evaluate_all('(nodes)=>nodes.map(n=>n.value).filter(Boolean)')
            assert len(subjects)==10 and len(concepts)==166
            page.wait_for_function('Array.from(document.querySelectorAll("img")).filter(i=>i.getBoundingClientRect().top<1000).every(i=>i.complete&&i.naturalWidth>0)')
            screenshot=output/'unblinded_first_stable_page.jpg';page.screenshot(path=str(screenshot),type='jpeg',quality=70)
            assert screenshot.stat().st_size<1024**2
            report['screenshots'].append(dict(path=str(screenshot),sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest()))
            report['checks'].append(dict(name='complete_population_initial_render',counter=initial,displayed_rows=40,subjects=10,concepts=166))
            first_id=first();page.get_by_role('button',name='下一页',exact=True).click()
            assert '第 2 / 415' in count() and first()!=first_id
            page.get_by_role('button',name='上一页',exact=True).click();assert first()==first_id
            page.get_by_role('button',name='上一页',exact=True).click();assert '第 1 / 415' in count()
            report['checks'].append(dict(name='next_previous_and_first_page_boundary',passed=True))
            page.locator('#subject').select_option(subjects[0]);assert count().startswith('1660 / 16600')
            assert all(x.startswith(subjects[0]+'/') for x in page.locator('#images code').all_text_contents())
            page.locator('#concept').select_option(concepts[0]);assert count().startswith('10 / 16600')
            assert page.locator('#images tr').count()==11
            query=first();page.locator('#search').fill(query);assert count().startswith('1 / 16600')
            report['checks'].append(dict(name='subject_concept_and_exact_query_filters',subject=subjects[0],concept=concepts[0],query=query,passed=True))
            page.locator('#search').fill('');page.locator('#concept').select_option('')
            for _ in range(41):page.get_by_role('button',name='下一页',exact=True).click()
            assert '第 42 / 42' in count() and page.locator('#images tr').count()==21
            page.get_by_role('button',name='下一页',exact=True).click();assert '第 42 / 42' in count()
            report['checks'].append(dict(name='subject_final_page_and_last_boundary',counter=count(),rows=20))
            page.locator('#subject').select_option('');assert count().startswith('16600 / 16600')
            target=page.locator('#images tr').nth(1).locator('a').first.get_attribute('href')
            response=context.request.get(origin+'/'+target);assert response.status==200 and response.headers['content-type'].startswith('image/')
            report['checks'].append(dict(name='full_target_link_serves_image',status=response.status))
            response=page.goto(origin+'/blinded/index.html',wait_until='networkidle',timeout=45000);assert response.status==200
            assert count().startswith('16600 / 16600') and page.locator('#images th').all_text_contents()==['目标 / ID','A']
            text=page.locator('body').inner_text()
            assert all(value not in text for value in ['REF_REG','DINO2AFC','SEED','ObjectF1'])
            page.locator('#subject').select_option(subjects[0]);assert count().startswith('1660 / 16600')
            report['checks'].append(dict(name='complete_blinded_population_and_hidden_methods_scores',passed=True))
            unexpected=[item for item in failed if 'ERR_ABORTED' not in str(item['error'])]
            assert not errors and not external and not unexpected,(errors,external,unexpected)
            report.update(status='PASSED_ACTUAL_BROWSER_UI_CHECKS',browser_version=browser.version,
                JS_errors=errors,external_request_attempts=external,failed_requests=unexpected,
                expected_navigation_or_filter_cancelled_requests=len(failed)-len(unexpected),
                standalone_file_protocol_opened=False,human_ratings_performed=False,new_idea_image_success=False,
                limits='Private HTTP browser UI verified. Complete counts/files were independently checked on server; human quality and standalone file-protocol execution were not tested.')
            browser.close();browser=None
    except Exception as error:
        report.update(status='FAILED_BROWSER_QA_PRESERVE_EVIDENCE',error=repr(error));raise
    finally:
        server.shutdown();server.server_close();client.close()
        report.update(seconds=time.monotonic()-started)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
