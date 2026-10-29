from fractions import Fraction as F
from itertools import combinations,product
from lrac.sparse_cues import (solve,required_support,lower_support_bound,cancellation_actions,
                              construct_continuous,hankel_rank)
from lrac.sparse_search import search,at_tolerance,transform


def test_pairwise_bound_holds_for_all_small_vectors_and_rotations():
    for x in product([F(-1),F(0),F(3,5),F(4,5)],repeat=3):
        for eta in [F(0),F(1,100),F(1,10),F(1)]:
            k=required_support(x,eta)["k"]
            for i,j in combinations(range(3),2):
                for sign in (-1,1):
                    y=transform(x,((i,j,sign),),inverse=True)
                    assert required_support(y,eta)["k"]>=k-1


def test_ratio_lookup_finds_exactly_the_support_reducing_actions():
    x=(F(3,5),F(4,5),F(-3,5),F(0))
    expected=[]
    for i,j in combinations(range(4),2):
        for sign in (-1,1):
            y=transform(x,((i,j,sign),),inverse=True)
            if sum(v!=0 for v in y)<sum(v!=0 for v in x):
                expected.append((i,j,sign))
    assert cancellation_actions(x)==sorted(expected)


def test_nonmonotone_trap_requires_looking_past_an_unhelpful_step():
    x=(F(-7,25),F(24,25),F(0))
    oracle=search(x,max_depth=2)
    assert at_tolerance(oracle,0,2)["k"]==1
    for method in ["ratio_greedy","tail_greedy"]:
        result=solve(x,0,method=method)
        assert result["k"]==2
        assert not result["certified"]
    for method in ["beam","branch_bound"]:
        result=solve(x,0,method=method)
        assert result["k"]==1 and result["certified"]


def test_bound_search_matches_exhaustive_oracle_at_each_tolerance():
    cases=[(F(0),)*3,(F(1),F(0),F(0)),(F(1,25),F(-9,25),F(7,25)),
           transform((1,2,0),((0,1,1),(1,2,-1)))]
    for x in cases:
        oracle=search(x,max_depth=2)
        for eta in [F(0),F(1,100),F(1,10),F(1)]:
            optimum=at_tolerance(oracle,eta,2)["k"]
            for method in ["ratio_greedy","tail_greedy","beam","branch_bound"]:
                result=solve(x,eta,method=method)
                assert result["k"]>=optimum
                assert result["error_sq"]<=eta*sum(v*v for v in x)
                if result["certified"]:
                    assert result["k"]==optimum
                if method=="branch_bound":
                    assert result["certified"] and result["k"]==optimum


def test_continuous_construction_attains_the_proven_lower_bound():
    for x in [(F(0),)*8,tuple(F(i-3,25) for i in range(8)),
              (F(-7,25),F(24,25))+(F(0),)*6]:
        energy=sum(v*v for v in x)
        for eta in [F(0),F(1,100),F(1,10),F(1)]:
            q=required_support(x,eta)["k"]
            for depth in range(8):
                result=construct_continuous(x,eta,depth)
                assert result["k"]==lower_support_bound(q,depth,energy,eta)
                assert result["k"]+result["angle_parameters"]==q
                assert abs(result["observed_error_sq"]-float(result["ideal_error_sq"]))<1e-12


def test_hankel_rank_exposes_recurrence_examples():
    assert hankel_rank(tuple(F(2**i,25) for i in range(8)))==1
    assert hankel_rank(tuple(F(i,25) for i in range(8)))==2
    assert hankel_rank((0,1,0,-1,0,1,0,-1))==2
    assert hankel_rank((F(3,5),F(12,25),F(16,25),2,0,0,0,0))==4
