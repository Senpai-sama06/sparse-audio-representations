"""Freeze experiment statuses and maintain a local index of research artifacts."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from approaches.parametric.portable_report import export_html
from approaches.parametric.progress import atomic_json
from util.provenance import file_hash


def register():
    runs=Path(__file__).resolve().parents[1]/'runs';archive=runs/'parametric_reports';archive.mkdir(exist_ok=True)
    definitions=[('multi_window_pilot','parametric_ablation_librispeech_1272_v2','preliminary_multi_window_sweep_stopped_after_audit.html'),
                 ('isolated_40ms_capacity','parametric_isolated_40ms_librispeech','isolated_40ms_capacity_complete.html')]
    records=[]
    for name,folder,report in definitions:
        source=runs/folder;status=json.loads((source/'progress.json').read_text());target=archive/report
        if name=='isolated_40ms_capacity':
            export_html(source/'index.html',target,title='Completed isolated 40 ms representation capacity study',
                        notice='Frozen completed experiment: 972/972 jobs, zero execution failures. Three isolated blocks; no stitching, quantization or RVQ. All audio, figures, progress and linked tables are embedded. Current Chrome/Edge: click Load embedded audio.')
        if not target.exists():raise FileNotFoundError(target)
        destination=archive/'records'/name;destination.mkdir(parents=True,exist_ok=True)
        snapshots={}
        for filename in ('progress.json','manifest.json','environment.json','metrics.csv','selections.csv','jobs.json'):
            if (source/filename).exists():
                shutil.copy2(source/filename,destination/filename);snapshots[filename]=file_hash(destination/filename)
        records.append(dict(id=name,status=status['status'],jobs_completed=status['jobs_completed'],jobs_total=status['jobs_total'],
                            elapsed_seconds=status['elapsed_seconds'],failures=status['failures'],run_directory=str(source),
                            report=report,report_bytes=target.stat().st_size,report_sha256=file_hash(target),snapshots=snapshots))
    active=runs/'parametric_learned_representation_pilot'
    if (active/'launch.json').exists():
        launch=json.loads((active/'launch.json').read_text(encoding='utf-8-sig'))
        status=json.loads((active/'progress.json').read_text()) if (active/'progress.json').exists() else {}
        records.append(dict(id='learned_representation_pilot',status=status.get('status','starting'),
                            jobs_completed=status.get('jobs_completed',0),jobs_total=status.get('jobs_total',0),
                            elapsed_seconds=status.get('elapsed_seconds',0),failures=status.get('failures',0),
                            run_directory=str(active),report=launch['report'],report_bytes=0,
                            live_progress=launch['dashboard'],note='Live experiment; counts are an index snapshot, follow live progress'))
    atomic_json(archive/'experiments.json',records)
    rows=''.join(f'<tr><td>{r["id"]}</td><td>{r["status"]}</td><td>{r["jobs_completed"]}/{r["jobs_total"]}</td><td>{r["elapsed_seconds"]/3600:.2f} h</td><td><a href="{r["report"]}">Self-contained report'+(f' ({r["report_bytes"]/1e6:.1f} MB)' if r['report_bytes'] else '')+'</a>'+ (f' · <a href="{r["live_progress"]}">Live progress</a>' if 'live_progress' in r else '')+'</td></tr>' for r in records)
    (archive/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Parametric experiment archive</title><style>body{font:16px system-ui;margin:3rem}td,th{padding:.7rem;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>Parametric experiment archive</h1><p>Original run directories retain caches, audio, payloads, logs and per-job results. Frozen status/metrics/settings snapshots and SHA-256 hashes are in records/ and experiments.json.</p><table><tr><th>Experiment</th><th>Status</th><th>Jobs</th><th>Active time</th><th>Report</th></tr>'+rows+'</table>',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='snapshots'} for r in records],indent=2))


if __name__=='__main__':register()
