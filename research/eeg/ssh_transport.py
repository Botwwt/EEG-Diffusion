import sys
sys.stdin.reconfigure(encoding='utf-8');sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0,r'C:\Users\Asus\AppData\Roaming\Python\Python311\site-packages')
import json,paramiko,pathlib,shlex
p=json.loads(sys.stdin.read())
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('CLOUD_LOGIN_HOST',port=54067,username='root',password=p['password'],timeout=25,banner_timeout=25)
try:
 if p['op']=='run':
  si,so,se=c.exec_command(p['cmd'],timeout=p.get('timeout',90));si.close()
  out=so.read().decode('utf-8',errors='replace');err=se.read().decode('utf-8',errors='replace')
  print(json.dumps({'code':so.channel.recv_exit_status(),'stdout':out,'stderr':err},ensure_ascii=False))
 elif p['op']=='put':
  s=c.open_sftp();s.put(p['local'],p['remote']);s.close();print(json.dumps({'uploaded':p['remote']}))
 elif p['op']=='get':
  s=c.open_sftp();s.get(p['remote'],p['local']);s.close();print(json.dumps({'downloaded':p['local']}))
finally:c.close()
