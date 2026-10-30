"""Resume an exhaustive search: python tools/resume_l0.py --run runs/YOUR_RUN."""
import argparse
from lrac.l0_parallel import resume_search_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=0, help="Zero uses all logical CPUs")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    print(resume_search_parallel(args.run, args.workers, args.chunk_size))
