"""Test structural cues against the exact sparse-domain optimizer; no training."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
from fractions import Fraction as F
import json
from pathlib import Path
import random
import statistics
import time
from lrac.sparse_search import search,at_tolerance,transform
from lrac.sparse_cues import solve,required_support,cancellation_actions,hankel_rank,construct_continuous
from lrac.utils import file_hash,write_json
from search_sparsest import examples


def dataset():
    cases=[dict(name=name,group="previous_examples",x=x) for name,x in examples().items()]
    cases.append(dict(name="two_step_trap",group="adversarial",
        x=(F(-7,25),F(24,25))+(F(0),)*6))
    cases.append(dict(name="geometric_growth",group="recurrence",x=tuple(F(2**i,25) for i in range(8))))
    cases.append(dict(name="linear_ramp",group="recurrence",x=tuple(F(i,25) for i in range(8))))
    rng=random.Random(20261029)
    actions=[(i,j,s) for i in range(8) for j in range(i+1,8) for s in (-1,1)]
    planted=[]
    for k in [1,2,3]:
        for index in range(16):
            z=[F(0)]*8
            for i in rng.sample(range(8),k):
                z[i]=F(rng.choice([-3,-2,-1,1,2,3]))
            x=tuple(z)
            word=[]
            for _ in range(2):
                candidates=[op for op in actions if x[op[0]] or x[op[1]]]
                op=rng.choice(candidates)
                word.append(op)
                x=transform(x,(op,))
            case=dict(name=f"planted_{k}_{index:02d}",group=f"planted_{k}",x=x,
                      generating_word=word,generating_code=z)
            planted.append(case)
            cases.append(case)
    for index in range(18):
        cases.append(dict(name=f"dense_{index:02d}",group="dense",
                          x=tuple(F(rng.randint(-25,25),25) for _ in range(8))))
    for index in range(20):
        parent=rng.choice(planted)
        noise=tuple(F(rng.choice([-1,0,1]),25) for _ in range(8))
        cases.append(dict(name=f"perturbed_{index:02d}",group="perturbed",
                          x=tuple(a+b for a,b in zip(parent["x"],noise)),parent=parent["name"]))
    assert len(cases)==96
    return cases


def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recurrence_demos(cases):
    byname={c["name"]:c["x"] for c in cases}
    result={}
    for name in ["geometric_growth","linear_ramp","harmonic_cycle"]:
        x=byname[name]
        if name=="geometric_growth":
            ratio=x[1]/x[0]
            parameters=(x[0],ratio)
            y=[x[0]]
            for _ in range(7):
                y.append(ratio*y[-1])
            rule="y[n+1] = ratio*y[n]"
        elif name=="linear_ramp":
            parameters=(x[0],x[1])
            y=[x[0],x[1]]
            for _ in range(6):
                y.append(2*y[-1]-y[-2])
            rule="y[n+2] = 2*y[n+1]-y[n]; rule coefficients are shared"
        else:
            alpha=(x[2]+x[0])/x[1]
            parameters=(x[0],x[1],alpha)
            y=[x[0],x[1]]
            for _ in range(6):
                y.append(alpha*y[-1]-y[-2])
            rule="y[n+2] = alpha*y[n+1]-y[n]"
        assert tuple(y)==x
        result[name]=dict(rule=rule,scalar_parameters_including_zeros=[str(v) for v in parameters],
            reconstruction_error_sq="0",hankel_rank=hankel_rank(x),
            scope="Different representation family; rule ID, precision, range and stability still require explicit accounting")
    return result


def main(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cases=dataset()
    etas=[F(0),F(1,100),F(1,10)]
    methods=["ratio_greedy","tail_greedy","beam","branch_bound"]
    rows,features,witnesses,continuous=[],[],{},[]
    started=time.perf_counter()
    oracle_seconds=0.
    for index,case in enumerate(cases):
        x,name=case["x"],case["name"]
        clock=time.perf_counter()
        oracle=search(x,max_depth=2)
        elapsed=time.perf_counter()-clock
        oracle_seconds+=elapsed
        features.append(dict(case=name,group=case["group"],hankel_rank=hankel_rank(x),
            initial_nonzeros=sum(v!=0 for v in x),exact_zero_actions=len(cancellation_actions(x))))
        case_witnesses=[]
        for eta in etas:
            optimal=at_tolerance(oracle,eta,2)
            root_k=required_support(x,eta)["k"]
            one_step=at_tolerance(oracle,eta,1)["k"]
            common=dict(case=name,group=case["group"],nmse_limit=str(eta),
                        oracle_k=optimal["k"],root_k=root_k,best_one_step_k=one_step)
            construction=construct_continuous(x,eta,2)
            if abs(construction["observed_error_sq"]-float(construction["ideal_error_sq"]))>1e-12:
                raise AssertionError("Continuous construction roundoff exceeds tolerance")
            continuous.append(dict(case=name,group=case["group"],nmse_limit=str(eta),
                fixed_angle_optimum=optimal["k"],continuous_angle_optimum=construction["k"],
                continuous_angles=construction["angle_parameters"],coefficient_values=construction["k"],
                scalar_values_plus_angles=construction["scalar_values_and_angles"],
                original_required_support=root_k,ideal_error_sq=str(construction["ideal_error_sq"]),
                observed_float64_error_sq=construction["observed_error_sq"]))
            rows.append(dict(**common,method="exhaustive",k=optimal["k"],regret=0,
                achieved_nmse=float(optimal["error_sq"]/oracle["energy"]) if oracle["energy"] else 0.,
                certified=True,candidate_states=3193,scored_states=3193,rotation_updates=6328,
                bound_pruned=0,duplicate_pruned=0,width_discarded=0,
                seconds_per_target_amortized=elapsed/len(etas)))
            for method in methods:
                result=solve(x,eta,method=method,max_depth=2,width=8)
                regret=result["k"]-optimal["k"]
                if regret<0 or (result["certified"] and regret!=0) or (method=="branch_bound" and regret):
                    raise AssertionError(f"Invalid bound/search result: {name}, {eta}, {method}")
                rows.append(dict(**common,method=method,k=result["k"],regret=regret,
                    achieved_nmse=float(result["error_sq"]/oracle["energy"]) if oracle["energy"] else 0.,
                    certified=result["certified"],**{k:result[k] for k in [
                        "candidate_states","scored_states","rotation_updates","bound_pruned","duplicate_pruned","width_discarded"]},
                    seconds_per_target_amortized=result["seconds"]))
                case_witnesses.append(dict(method=method,nmse_limit=str(eta),k=result["k"],
                    word=[list(op) for op in result["word"]],z=[str(v) for v in result["z"]],
                    error_sq=str(result["error_sq"]),lower_bound=result["lower_bound"],
                    upper_bound=result["upper_bound"],certified=result["certified"]))
        witnesses[name]=dict(group=case["group"],sequence=[str(v) for v in x],solutions=case_witnesses)
        if (index+1)%8==0:
            print(f"CUES {index+1}/{len(cases)}",flush=True)
    summary_rows=[]
    for eta in etas:
        for group in ["all",*sorted({c["group"] for c in cases})]:
            for method in ["exhaustive",*methods]:
                subset=[r for r in rows if r["nmse_limit"]==str(eta) and r["method"]==method
                        and (group=="all" or r["group"]==group)]
                summary_rows.append(dict(group=group,nmse_limit=str(eta),method=method,cases=len(subset),
                    optimal_fraction=sum(r["regret"]==0 for r in subset)/len(subset),
                    mean_support_regret=statistics.mean(r["regret"] for r in subset),
                    max_support_regret=max(r["regret"] for r in subset),
                    certified_fraction=sum(r["certified"] for r in subset)/len(subset),
                    median_candidate_states=statistics.median(r["candidate_states"] for r in subset),
                    mean_candidate_states=statistics.mean(r["candidate_states"] for r in subset),
                    median_rotation_updates=statistics.median(r["rotation_updates"] for r in subset)))
    write_csv(output/"sparse_cues_results.csv",rows)
    write_csv(output/"sparse_cues_summary.csv",summary_rows)
    write_csv(output/"sparse_cues_features.csv",features)
    write_csv(output/"sparse_cues_continuous.csv",continuous)
    meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=20261029,cases=len(cases),
        targets=[str(v) for v in etas],domain="Same 8-dimensional at-most-two rotation family as sparse_search",
        coefficient_grid=625,coefficient_bound=8,beam_width=8,trained_models=False,
        comparison_note="Exhaustive oracle computes all budgets/targets together; its measured time is divided among three targets. State/rotation counts are per-target reference work, not total actual replay counts.",
        certificate_note="Heuristics certify only on matching the proven root lower bound; branch_bound also certifies by exhausting all unpruned improving states. Post-hoc oracle agreement is a separate metric.",
        cue_scope="Finite rational rotation geometry, not established acoustic features",
        recurrence_examples=recurrence_demos(cases),oracle_seconds=oracle_seconds,
        continuous_note="A different family: up to two arbitrary real-angle pair rotations, unrestricted real coefficients. Ideal optimum proved; float64 demonstration checked to absolute squared-error tolerance1e-12. Angles, pair indices and precision are not free.",
        seconds=time.perf_counter()-started,
        sources={p:file_hash(Path(p)) for p in ["lrac/sparse_search.py","lrac/sparse_cues.py","tools/analyze_sparse_cues.py"]},
        witnesses=witnesses)
    write_json(output/"sparse_cues_results.json",meta)
    print(json.dumps([r for r in summary_rows if r["group"]=="all"],indent=2),flush=True)
    print(f"COMPLETE {len(rows)} comparisons in {meta['seconds']:.2f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="documentation")
    main(parser.parse_args())
