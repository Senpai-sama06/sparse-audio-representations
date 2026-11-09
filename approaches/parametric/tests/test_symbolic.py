"""Checks that distinguish formula discovery, coefficient fitting and storage."""
import json
import numpy as np
import pytest
import torch
from approaches.parametric.symbolic.expressions import evaluate,validate,description,fit_expression
from approaches.parametric.symbolic.model import ExpressionGenerator,random_expression,synthetic_example
from approaches.parametric.symbolic.experiment import known_answers
from approaches.parametric.symbolic.payload import reconstruct
from approaches.parametric.symbolic.search import search


def test_known_structure_chirp_fits_and_jacobian_is_correct():
    checks=known_answers()
    assert checks['chirped_amfm_nmse']<1e-12
    assert checks['jacobian_max_error']<1e-7


@pytest.mark.parametrize('tokens',[[],['+','t'],['t','C'],['__import__']])
def test_invalid_program_rejected(tokens):
    with pytest.raises(ValueError):validate(tokens)


def test_generated_programs_respect_grammar_and_variable_budgets():
    torch.manual_seed(1);torch.set_num_threads(1);model=ExpressionGenerator(hidden=16)
    rng=np.random.default_rng(4)
    for budget in (7,15,31):
        programs=model.propose(np.zeros(640),budget,12,rng)
        programs += [random_expression(rng,budget) for _ in range(12)]
        for tokens in programs:validate(tokens);assert len(tokens)<=budget
        lp=model.log_prob(torch.zeros(len(programs),640),programs,torch.full((len(programs),),float(budget)))
        assert torch.isfinite(lp).all()
        (-lp.mean()).backward();model.zero_grad()


def test_explicit_formula_reconstructs_without_a_model_or_samples(tmp_path):
    tokens=['*','C','sin','*','C','t'];constants=[.2,50.]
    counts=description(tokens)
    assert counts['parameters']==3  # amplitude, frequency, explicit output gain
    assert counts['description_bits']==16+4*len(tokens)+3*64
    path=tmp_path/'expression.json'
    path.write_text(json.dumps(dict(format='symbolic-imf-v1',tokens=tokens,constants=constants,gain=.5,samples=640,sample_rate=16000,parameters=3)))
    y,rate=reconstruct(path)
    assert rate==16000 and np.allclose(y,.1*np.sin(50*np.linspace(-1,1,640)))


def test_search_budget_counts_duplicate_proposals():
    y=np.sin(10*np.linspace(-1,1,640));result=search(y,None,seed=3,candidates=9,budgets=(1,),max_nfev=5,restarts=1,refine=0)
    assert result['proposals']==9 and result['unique']<=2
    assert len(result['records'])==result['unique']


def test_synthetic_curriculum_is_finite_and_not_all_one_template():
    rng=np.random.default_rng(17);examples=[synthetic_example(rng) for _ in range(20)]
    assert all(np.isfinite(y).all() and np.isclose(np.mean(y*y),1,rtol=1e-5) for y,_ in examples)
    assert len({tuple(tokens) for _,tokens in examples})>10


def test_output_gain_is_fitted_even_without_internal_constants():
    t=np.linspace(-1,1,640);y=-.2*np.sin(t)
    fitted=fit_expression(['sin','t'],y)
    assert fitted.nmse<1e-25
    assert np.isclose(fitted.gain,-.2) and np.allclose(fitted.waveform(t),y)
