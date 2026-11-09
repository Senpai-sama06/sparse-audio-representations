# Parametric approaches and reports

Published reports live in a folder named `report` within each study subfolder.
Raw run data, checkpoints, payloads, caches, and live incremental reports remain
under the original `runs/parametric_*` directories.

| Study | Report | Saved status |
| --- | --- | --- |
| Multi-window parametric fitting | [Report](multi_window/report/preliminary_multi_window_sweep_stopped_after_audit.html) | Stopped after audit, 452/744 jobs; about 591 MB |
| Isolated 40 ms curve fitting | [Report](isolated_40ms/report/isolated_40ms_capacity_complete.html) | Complete, 972/972 jobs; about 14.2 MB |
| Learned shared representations | [Report](shared_representation/report/shared_representation_pilot_complete.html) | Completed pilot; about 3.7 MB |
| Explicit IMF expression discovery | [Report](symbolic/report/symbolic_imf_pilot_complete.html) | Completed pilot; about 9.3 MB |

The studies answer different questions. The shared-representation pilot learns
neural waveform representations; the symbolic pilot discovers explicit
functions for individual intrinsic mode functions (IMFs). The stopped
multi-window report is incomplete evidence and is retained with that label.

## Combined results and reading material

- [Archive index](report/index.html)
- [Standalone combined results overview](report/experiment_results_2026_11_04/experiment_results_shareable.html)
- [Linked overview with separate assets and evidence](report/experiment_results_2026_11_04/index.html)
- [Focused reading guide](report/experiment_results_2026_11_04/reading_guide.html)
- [Annotated literature review](report/literature_review_2026_11_04/index.html)
- [Local-paper reading discussion](report/local_paper_reading_2026_11_04/discussion.html)
- [Saved experiment ledger](report/experiments.json)
- [Relocation and integrity record](report/relocation.json)

The four study HTML reports and combined shareable overview were relocated
without changing their bytes. They embed their playback assets. In a current
Chrome or Edge browser, use **Load embedded audio** where shown. The linked
overview needs its adjacent `audio/`, images, and tables. Reading guides link
to the shared PDF library under `documentation/literature/`.

The `records/` folders beside the multi-window, isolated, and shared reports
retain their original frozen metrics, progress, settings, and provenance.
The historical archive did not include a separate symbolic snapshot folder;
its report remains self-contained and its original run is retained.

## Publication commands

Run commands from the repository root:

```powershell
python tools/register_parametric_runs.py
python tools/write_parametric_experiment_report.py
python tools/write_parametric_literature_review.py
```

The registration script updates the index and ledger, snapshots the configured
local runs, and publishes reports to the study-specific `report/` folders. It
preserves existing frozen HTML files. The retrospective and literature tools
write to this approach's shared `report/` folder. These commands require their
original inputs; relocation itself does not run experiments or retrain models.

The report layout is defined in [report_paths.py](report_paths.py). Method code
for multi-window, isolated, and shared studies remains in [ablation.py](ablation.py),
[isolated.py](isolated.py), and [learned.py](learned.py). Symbolic code is under
[symbolic/](symbolic/). The new publication subfolders do not change those Python
module paths.

The original archive ledger and paper inventory are preserved as
`experiments_before_relocation.json` and `inventory_before_relocation.json`
beside their updated versions. Historical run paths and source hashes inside
frozen reports and source ledgers describe the original builds; the relocation
record maps moved files to their current locations.
