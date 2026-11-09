"""Published parametric reports; raw experiment products remain under runs/."""
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
ARCHIVE = METHOD_ROOT / "report"
REPORT_DIRECTORIES = {
    "multi_window_pilot": METHOD_ROOT / "multi_window" / "report",
    "isolated_40ms_capacity": METHOD_ROOT / "isolated_40ms" / "report",
    "learned_representation_pilot": METHOD_ROOT / "shared_representation" / "report",
    "symbolic_imf_discovery": METHOD_ROOT / "symbolic" / "report",
}
REPORT_FILENAMES = {
    "multi_window_pilot": "preliminary_multi_window_sweep_stopped_after_audit.html",
    "isolated_40ms_capacity": "isolated_40ms_capacity_complete.html",
    "learned_representation_pilot": "shared_representation_pilot_complete.html",
    "symbolic_imf_discovery": "symbolic_imf_pilot_complete.html",
}
