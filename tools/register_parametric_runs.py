"""Freeze experiment statuses and maintain a local index of research artifacts."""
import json
import hashlib
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from approaches.parametric.portable_report import export_html
from approaches.parametric.progress import atomic_json
from approaches.parametric.report_paths import ARCHIVE, REPORT_DIRECTORIES, REPORT_FILENAMES


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_relative(path):
    return os.path.relpath(path, ARCHIVE).replace("\\", "/")


def write_index(records):
    """Link study-owned publications without regenerating frozen reports."""
    import html
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        link = html.escape(record['report'], quote=True)
        rows.append('<tr><td>' + html.escape(record['id']) + '</td><td>' +
                    html.escape(record['status']) + '</td><td>' +
                    f"{record['jobs_completed']}/{record['jobs_total']}" +
                    f"</td><td>{record['elapsed_seconds']/3600:.2f} h</td>" +
                    f'<td><a href="{link}">Report ({record["report_bytes"]/1e6:.1f} MB)</a></td></tr>')
    document = ['<!doctype html><html lang="en"><meta charset="utf-8">',
                '<title>Parametric experiment archive</title><h1>Parametric experiment archive</h1>',
                '<p>Each experiment owns a report/ folder. Original run directories retain caches, audio, payloads, checkpoints, logs, and per-job results. Saved statuses and report hashes are in experiments.json; snapshots sit beside each study report.</p>',
                '<table border="1"><tr><th>Experiment</th><th>Status</th><th>Jobs</th><th>Active time</th><th>Report</th></tr>',
                *rows, '</table>']
    for relative, label in (
        ('parametric_experiments_testcases.html', 'Parametric test cases: PitchFlower-style comparison tables'),
        ('parametric_experiments_tabbed.html', 'All experiments: single tabbed report with source signals and VMD context'),
        ('experiment_results_2026_11_04/experiment_results_shareable.html', 'Combined results: standalone overview'),
        ('experiment_results_2026_11_04/reading_guide.html', 'Focused reading guide'),
        ('literature_review_2026_11_04/index.html', 'Annotated literature review'),
        ('local_paper_reading_2026_11_04/discussion.html', 'Local-paper reading discussion'),
    ):
        if (ARCHIVE/relative).is_file():
            document.append(f'<p><a href="{relative}">{label}</a></p>')
    document.append('</html>')
    (ARCHIVE/'index.html').write_text('\n'.join(document), encoding='utf-8', newline='\n')


def register():
    runs=Path(__file__).resolve().parents[1]/'runs';archive=ARCHIVE;archive.mkdir(parents=True,exist_ok=True)
    definitions=[('multi_window_pilot','parametric_ablation_librispeech_1272_v2','preliminary_multi_window_sweep_stopped_after_audit.html'),
                 ('isolated_40ms_capacity','parametric_isolated_40ms_librispeech','isolated_40ms_capacity_complete.html')]
    records=[]
    for name,folder,report in definitions:
        source=runs/folder;status=json.loads((source/'progress.json').read_text());target=REPORT_DIRECTORIES[name]/report
        if name=='isolated_40ms_capacity' and not target.exists():
            export_html(source/'index.html',target,title='Completed isolated 40 ms representation capacity study',
                        notice='Frozen completed experiment: 972/972 jobs, zero execution failures. Three isolated blocks; no stitching, quantization or RVQ. All audio, figures, progress and linked tables are embedded. Current Chrome/Edge: click Load embedded audio.')
        if not target.exists():raise FileNotFoundError(target)
        destination=REPORT_DIRECTORIES[name]/'records';destination.mkdir(parents=True,exist_ok=True)
        snapshots={}
        for filename in ('progress.json','manifest.json','environment.json','metrics.csv','selections.csv','jobs.json'):
            if (source/filename).exists():
                shutil.copy2(source/filename,destination/filename);snapshots[filename]=file_hash(destination/filename)
        records.append(dict(id=name,status=status['status'],jobs_completed=status['jobs_completed'],jobs_total=status['jobs_total'],
                            elapsed_seconds=status['elapsed_seconds'],failures=status['failures'],run_directory=str(source),
                            report=archive_relative(target),records_directory=archive_relative(destination),report_bytes=target.stat().st_size,report_sha256=file_hash(target),snapshots=snapshots))
    active=runs/'parametric_learned_representation_pilot'
    if (active/'launch.json').exists():
        launch=json.loads((active/'launch.json').read_text(encoding='utf-8-sig'))
        status=json.loads((active/'progress.json').read_text()) if (active/'progress.json').exists() else {}
        completed=status.get('status')=='complete'
        frozen=REPORT_DIRECTORIES['learned_representation_pilot']/REPORT_FILENAMES['learned_representation_pilot']
        frozen.parent.mkdir(parents=True,exist_ok=True)
        snapshots={}
        if completed:
            # Preserve the original report byte-for-byte. This exploratory run
            # learned shared waveform representations, not IMF expressions.
            if not frozen.exists():shutil.copy2(active/'learned_representation_shareable.html',frozen)
            destination=REPORT_DIRECTORIES['learned_representation_pilot']/'records';destination.mkdir(parents=True,exist_ok=True)
            for filename in ('progress.json','manifest.json','environment.json','metrics.csv','assets.json','history.json','training.csv','data_manifest.json'):
                shutil.copy2(active/filename,destination/filename);snapshots[filename]=file_hash(destination/filename)
        records.append(dict(id='learned_representation_pilot',status=status.get('status','starting'),
                            jobs_completed=status.get('jobs_completed',0),jobs_total=status.get('jobs_total',0),
                            elapsed_seconds=status.get('elapsed_seconds',0),failures=status.get('failures',0),
                            run_directory=str(active),report=archive_relative(frozen) if completed else launch['report'],
                            records_directory=archive_relative(frozen.parent/'records') if completed else None,
                            report_bytes=frozen.stat().st_size if completed else 0,
                            report_sha256=file_hash(frozen) if completed else None,snapshots=snapshots,
                            live_progress=launch['dashboard'],note='Exploratory shared representation pilot; does not discover explicit per-IMF functions.'))
    active=runs/'parametric_symbolic_imf_pilot'
    if (active/'progress.json').exists():
        status=json.loads((active/'progress.json').read_text())
        report=active/'symbolic_imf_shareable.html'
        frozen=REPORT_DIRECTORIES['symbolic_imf_discovery']/REPORT_FILENAMES['symbolic_imf_discovery']
        frozen.parent.mkdir(parents=True,exist_ok=True)
        completed=status['status']=='complete'
        if completed and not frozen.exists():shutil.copy2(report,frozen)
        records.append(dict(id='symbolic_imf_discovery',status=status['status'],jobs_completed=status['jobs_completed'],
                            jobs_total=status['jobs_total'],elapsed_seconds=status['elapsed_seconds'],failures=status['failures'],
                            run_directory=str(active),report=archive_relative(frozen) if completed else 'http://127.0.0.1:8770/symbolic_imf_shareable.html',
                            report_bytes=frozen.stat().st_size if completed else (report.stat().st_size if report.exists() else 0),
                            report_sha256=file_hash(frozen) if completed else None,live_progress='http://127.0.0.1:8770/'))
    atomic_json(archive/'experiments.json',records)
    write_index(records)
    print(json.dumps([{k:v for k,v in r.items() if k!='snapshots'} for r in records],indent=2))


if __name__=='__main__':register()
