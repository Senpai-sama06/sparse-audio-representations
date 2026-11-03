"""Local listening report for a single-testcase window sweep."""
import html


def write_report(destination, rows):
    """Write a self-contained comparison table linking the saved FLOAT WAVs."""
    columns = ["method", "window_ms", "reconstruction", "mean_modes", "pesq_wb", "stoi",
               "nmse", "snr_db", "si_sdr_db", "decomposition_rtf", "pesq_wb_status", "stoi_status"]

    def cell(value):
        if value is None:
            return "—"
        return html.escape(f"{value:.6g}" if isinstance(value, float) else str(value))

    body = []
    for row in rows:
        cells = "".join(f"<td>{cell(row.get(key))}</td>" for key in columns)
        audio = html.escape(row["audio"], quote=True)
        body.append(f'<tr>{cells}<td><audio controls preload="none" src="{audio}"></audio></td></tr>')
    header = "".join(f"<th>{html.escape(key)}</th>" for key in columns) + "<th>Listen</th>"
    content = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Parametric: EMD/VMD window sweep</title>
<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ccc;text-align:left}th{background:#eee}audio{width:220px}</style>
<h1>EMD/VMD window reconstruction</h1>
<p>Compare full-clip audio against the self-reference. PESQ-WB is a speech-quality
proxy; STOI measures intelligibility. Missing metric statuses are shown explicitly.</p>
<p><b>Scope:</b> sampled decomposition and stitching only. No parameter fitting or compression.
The residual control restores the native EMD residue or the VMD input-minus-mode-sum
residual; a high score there does not demonstrate compact coding. Mode identities
are not matched across frames. Fit time is shared by both reconstructions.</p>
<p><a href="metrics.csv">Metrics CSV</a> · <a href="settings.json">Settings</a> · <a href="summary.json">Summary</a></p>
'''
    (destination / "index.html").write_text(content + f"<table><thead><tr>{header}</tr></thead><tbody>" + "".join(body) + "</tbody></table></html>", encoding="utf-8")
