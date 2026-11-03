"""CLI arguments shared by main.py and the parametric package entry point."""
import argparse


def add_arguments(parser):
    parser.add_argument("--audio", required=True, help="Local mono speech testcase")
    parser.add_argument("--window-ms", type=float, nargs="+", default=[20, 40, 80, 160])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--emd-max-imfs", type=int, default=-1, help="-1 retains all extracted IMFs")
    parser.add_argument("--emd-max-iterations", type=int, default=1000)
    parser.add_argument("--vmd-modes", type=int, default=8)
    parser.add_argument("--vmd-alpha", type=float, default=2000.0)
    parser.add_argument("--vmd-tau", type=float, default=0.0)
    parser.add_argument("--vmd-tolerance", type=float, default=1e-7)
    parser.add_argument("--output", help="New run directory; existing directories are rejected")
    parser.add_argument("--save-modes", action="store_true", help="Save sampled modes/residue per frame for inspection")
    parser.add_argument("--no-speech-metrics", action="store_true", help="Disable PESQ/STOI explicitly")


def run_from_args(args):
    from approaches.parametric.experiment import run
    return run(args.audio, window_ms=args.window_ms, overlap=args.overlap,
               sample_rate=args.sample_rate, start_seconds=args.start_seconds,
               duration_seconds=args.duration_seconds, emd_max_imfs=args.emd_max_imfs,
               emd_max_iterations=args.emd_max_iterations, vmd_modes=args.vmd_modes,
               vmd_alpha=args.vmd_alpha, vmd_tau=args.vmd_tau, vmd_tolerance=args.vmd_tolerance,
               output=args.output, save_modes=args.save_modes, speech_metrics=not args.no_speech_metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EMD/VMD speech reconstruction over window lengths")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))
