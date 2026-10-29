from fractions import Fraction as F
from itertools import product
import pytest
from lrac.sparse_search import transform,words,search,at_tolerance


def test_inverse_and_energy_hold_in_exact_arithmetic():
    x=(F(2,25),F(-7,25),F(3,5))
    word=((0,1,1),(1,2,-1),(0,2,1))
    y=transform(x,word)
    assert transform(y,word,inverse=True)==x
    assert sum(v*v for v in y)==sum(v*v for v in x)


def test_analytic_search_matches_exhaustive_quantized_codes():
    # Independently enumerate EVERY coefficient pair and support in this small
    # finite alphabet; compare with analytic coefficient elimination over domains.
    x=(F(3,5),F(4,5))
    result=search(x,max_depth=1,coefficient_grid=25,coefficient_bound=1)
    alphabet=[F(i,25) for i in range(-25,26)]
    brute=[None]*3
    for word in words(2,1):
        for z in product(alphabet,repeat=2):
            y=transform(z,word)
            error=sum((a-b)**2 for a,b in zip(x,y))
            size=sum(v!=0 for v in z)
            for k in range(size,3):
                if brute[k] is None or error<brute[k]:
                    brute[k]=error
    assert [r["error_sq"] for r in result["curves"][1]]==brute
    witness=at_tolerance(result,F(0),1)
    assert witness["k"]==witness["certified_lower_bound"]==witness["certified_upper_bound"]==1
    assert witness["smaller_support_min_error_sq"]>witness["target_error_sq"]


def test_zero_and_zero_code_tolerance_boundary():
    zero=search((0,0),max_depth=1,coefficient_grid=25,coefficient_bound=1)
    assert at_tolerance(zero,0,1)["k"]==0
    signal=search((F(3,5),F(4,5)),max_depth=1,coefficient_grid=25,coefficient_bound=1)
    assert at_tolerance(signal,1,1)["k"]==0
    assert at_tolerance(signal,F(99,100),1)["k"]==1


def test_joint_search_finds_a_domain_requiring_two_rotations():
    x=transform((1,0,0),((0,1,1),(1,2,1)))
    result=search(x,max_depth=2)
    assert result["domain_words"]==1+6+36
    assert [at_tolerance(result,0,d)["k"] for d in range(3)]==[3,2,1]
    for d,curve in enumerate(result["curves"]):
        assert all(curve[k+1]["error_sq"]<=curve[k]["error_sq"] for k in range(3))
        if d:
            assert all(curve[k]["error_sq"]<=result["curves"][d-1][k]["error_sq"] for k in range(4))


def test_search_fails_when_analytic_projection_is_not_in_the_alphabet():
    with pytest.raises(ValueError,match="alphabet"):
        search((F(1,7),0),max_depth=1,coefficient_grid=25,coefficient_bound=1)
    with pytest.raises(TypeError,match="binary floats"):
        search((.6,.8))
