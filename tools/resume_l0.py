"""Resume an exhaustive search: python tools/resume_l0.py --run runs/YOUR_RUN."""
import argparse
from lrac.l0 import resume_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(resume_search(args.run))
