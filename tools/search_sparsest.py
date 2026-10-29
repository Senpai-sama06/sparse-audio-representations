"""Run a tiny exact optimization of domain choice and sparse coefficients."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import json
from math import ceil,log2
from pathlib import Path
import random
import time
from lrac.sparse_search import transform,search,at_tolerance
from lrac.utils import file_hash,write_json


def examples():
    impulse = (F(1),)+(F(0),)*7
    two = (F(1),F(0),F(0),F(2))+(F(0),)*4
    rng = random.Random(0)
    return dict(
        zero=(F(0),)*8,
        impulse=impulse,
        rotated_one=transform(impulse,((0,1,1),)),
        rotated_two=transform(two,((0,1,1),(1,2,1))),
        dense_rational=tuple(F(v,25) for v in [25,19,-11,7,5,-13,17,23]),
        harmonic_cycle=tuple(F(v) for v in [0,1,0,-1,0,1,0,-1]),
        noise_rational=tuple(F(rng.randint(-25,25),25) for _ in range(8)))


def serial(entry):
    return dict(k=entry["k"],word=[list(a) for a in entry["word"]],
        coefficients=[str(v) for v in entry["z"]],squared_error=str(entry["error_sq"]),
        target_squared_error=str(entry["target_error_sq"]),
        smaller_support_min_squared_error=str(entry["smaller_support_min_error_sq"])
            if entry["smaller_support_min_error_sq"] is not None else None,
        lower_bound=entry["certified_lower_bound"],upper_bound=entry["certified_upper_bound"])


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    n,depth,grid,bound=8,2,625,8
    actions=n*(n-1)
    action_bits=ceil(log2(actions))
    length_bits=ceil(log2(depth+1))
    tolerances=[F(0),F(1,1000),F(1,100),F(1,10),F(1,4)]
    records,curves,all_witnesses=[],[],{}
    started=time.perf_counter()
    for name,x in examples().items():
        clock=time.perf_counter()
        result=search(x,max_depth=depth,coefficient_grid=grid,coefficient_bound=bound)
        witnesses=[]
        for d in range(depth+1):
            for k,e in enumerate(result["curves"][d]):
                curves.append(dict(signal=name,max_rotation_depth=d,support_budget=k,
                    best_error_sq=str(e["error_sq"]),
                    best_nmse=float(e["error_sq"]/result["energy"]) if result["energy"] else 0.,
                    witness_word=json.dumps(e["word"]),actual_nonzeros=e["nonzeros"]))
            for eta in tolerances:
                entry=at_tolerance(result,eta,d)
                if entry is None:
                    raise AssertionError("Identity should make every example exactly feasible")
                witness=serial(entry)
                witnesses.append(dict(max_rotation_depth=d,nmse_limit=str(eta),**witness))
                records.append(dict(signal=name,nmse_limit=str(eta),max_rotation_depth=d,
                    nonzeros=entry["k"],achieved_nmse=float(entry["error_sq"]/result["energy"]) if result["energy"] else 0.,
                    theta_bits=length_bits+action_bits*len(entry["word"]),
                    rotations=len(entry["word"]),lower_bound=entry["certified_lower_bound"],
                    upper_bound=entry["certified_upper_bound"],certified=True))
        all_witnesses[name]=dict(sequence=[str(v) for v in x],energy=str(result["energy"]),
            domain_words=result["domain_words"],seconds=time.perf_counter()-clock,solutions=witnesses)
        exact=[r["nonzeros"] for r in records if r["signal"]==name and r["nmse_limit"]=="0"]
        print(f"{name}: exact minimum support by depth 0/1/2 = {exact}",flush=True)
    for filename,rows in [("sparse_search_results.csv",records),("sparse_search_curves.csv",curves)]:
        with (output/filename).open("w",newline="",encoding="utf-8") as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Exact global optimization ONLY over the declared finite rational rotation-word family and coefficient alphabet",
        dimension=n,max_rotation_depth=depth,primitive_rotations=actions,
        domain_words=sum(actions**d for d in range(depth+1)),
        domain_note="Words may describe the same matrix; duplicates do not invalidate exhaustive coverage",
        rotation_cosine="3/5",rotation_sine="+/-4/5",
        transform_description=dict(action_bits=action_bits,length_bits=length_bits,max_bits=length_bits+depth*action_bits),
        coefficient_alphabet=dict(step="1/625",lower="-8",upper="8",symbols=2*bound*grid+1),
        tolerances=[str(v) for v in tolerances],distortion="Squared waveform error <= eta * input energy; zero input reconstructs exactly",
        certificate="Exhaust all domain words; use exact orthogonal best-k tail energies; all minimizing coefficients verified inside the declared finite alphabet; Fraction arithmetic",
        trained_models=False,seconds=time.perf_counter()-started,
        source_sha256={p:file_hash(Path(p)) for p in ["lrac/sparse_search.py","tools/search_sparsest.py"]},
        examples=all_witnesses)
    write_json(output/"sparse_search_results.json",summary)
    print(f"Certified {len(records)} results; {summary['domain_words']} domain words per signal; {summary['seconds']:.2f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="documentation")
    main(parser.parse_args())
