"""Atomic progress, heartbeat and a dependency-free localhost dashboard.

Use: python -m approaches.parametric.progress RUN_DIRECTORY [--port 8765]
The HTTP endpoint /progress.json exposes the same live state as the dashboard.
"""
import json
import os
import threading
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers/antivirus can briefly hold the destination without delete
    # sharing. Retry replacement, preserving the old complete JSON meanwhile.
    for attempt in range(40):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.025)


class Progress:
    def __init__(self, directory, total, completed=0):
        self.path = Path(directory)/'progress.json'
        self.started = time.time(); self.previous = 0
        if self.path.exists():
            old = json.loads(self.path.read_text()); self.previous = old.get('elapsed_seconds',0)
        self.state = dict(status='running',pid=os.getpid(),jobs_total=total,jobs_completed=completed,
                          current_job=None,stage=None,frames_completed=0,frames_total=0,failures=0)
        self.durations = old.get('seconds_per_job_by_stage',{}) if self.previous else {}
        self.lock = threading.Lock(); self.stop = threading.Event(); self.last_write = 0
        self.thread = threading.Thread(target=self._heartbeat,daemon=True); self.thread.start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            if set(values) != {'frames_completed'} or time.time()-self.last_write > .5:
                self._write()

    def _write(self):
        now = time.time(); elapsed = self.previous+now-self.started
        self.last_write = now
        self.state.update(elapsed_seconds=elapsed,heartbeat_unix=now,
                          percent=100*self.state['jobs_completed']/max(1,self.state['jobs_total']))
        # Per-stage timings will be filled as comparable jobs finish. Unknown
        # stages keep ETA provisional rather than showing a spurious certainty.
        self.state['seconds_per_job_by_stage'] = self.durations
        self.state['eta_note'] = 'Provisional: job cost varies by decomposition, family and window; excludes pauses.'
        atomic_json(self.path,self.state)

    def _heartbeat(self):
        while not self.stop.wait(3):
            with self.lock:
                self._write()

    def close(self,status):
        self.stop.set(); self.thread.join(); self.update(status=status)


def serve(directory,port):
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    root = Path(directory).resolve()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,directory=str(root),**kwargs)
        def do_GET(self):
            if self.path == '/':
                page = '''<!doctype html><meta charset="utf-8"><title>Parametric progress</title>
<style>body{font:18px system-ui;margin:3rem}pre{font-size:14px}progress{width:90%}</style>
<h1>Parametric ablation progress</h1><progress id="bar" max="100"></progress><p id="summary"></p>
<p><a href="index.html">Results and listening</a> | <a href="metrics.csv">CSV</a></p><pre id="detail"></pre>
<script>async function refresh(){try{let p=await(await fetch('progress.json?'+Date.now())).json();
bar.value=p.percent;let age=Date.now()/1000-p.heartbeat_unix;
summary.textContent=`${p.status} | ${p.jobs_completed}/${p.jobs_total} jobs | elapsed ${(p.elapsed_seconds/3600).toFixed(2)} h | ETA ${p.eta_seconds==null?'calibrating':(p.eta_seconds/3600).toFixed(2)+' h'} | heartbeat ${age.toFixed(0)} s ago`;
detail.textContent=JSON.stringify(p,null,2);}catch(e){summary.textContent=String(e)}}refresh();setInterval(refresh,3000)</script>'''
                self.send_response(200); self.send_header('Content-type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(page.encode())
            else:
                super().do_GET()
        def log_message(self,*args):
            pass
    print(f'Progress: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('directory'); parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(); serve(args.directory,args.port)
