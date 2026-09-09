import copy
import random

import pytest
import torch

from jobs import ember_rung0_trajectory as trace


def test_schedule_matches_original_ladder_draw_order():
    cfg = trace.load_config()
    rng = random.Random(cfg['seed'])
    expected = []
    for _ in range(40):
        placement = [rng.randrange(24) for _ in range(4)]
        tool = [rng.randrange(110) for _ in range(8)]
        copied = [rng.randrange(40) for _ in range(8)]
        expected.append({'placement': placement, 'tool': tool, 'copy': copied})
    assert trace.batch_schedule(cfg, 24, 110, 40) == expected


def test_instrumented_steps_preserve_original_optimizer_trajectory(monkeypatch):
    torch.manual_seed(17)
    student = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(student)
    teacher = copy.deepcopy(student)
    examples = torch.tensor([[.1, .3, -.4], [.7, -.2, .6], [-.8, .4, .2]])
    target = torch.tensor([[.7, .1], [.2, .5], [.9, -.3]])
    def placement(model, _torch, _examples, indices):
        return (model(examples[indices]) - target[indices]).square().mean()
    def kl(model, frozen, _tokenizer, _torch, _rows, indices, _temperature):
        return (model(examples[indices]) - frozen(examples[indices]).detach()).square().mean()
    monkeypatch.setattr(trace.objectives, 'batch_loss', placement)
    monkeypatch.setattr(trace.distill, 'batch_teacher_kl', kl)
    cfg = trace.load_config()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-7, weight_decay=.01)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-7, weight_decay=.01)
    rng = random.Random(cfg['seed'])
    for batch in trace.batch_schedule(cfg, 3, 3, 3):
        trace.optimizer_step(student, teacher, None, torch, cfg, [], [], [], batch, opt)
        ref_opt.zero_grad(set_to_none=True)
        place_idx = [rng.randrange(3) for _ in range(4)]
        tool_idx = [rng.randrange(3) for _ in range(8)]
        copy_idx = [rng.randrange(3) for _ in range(8)]
        total = (.20 * placement(reference, torch, [], place_idx)
                 + .55 * kl(reference, teacher, None, torch, [], tool_idx, 1.)
                 + .25 * kl(reference, teacher, None, torch, [], copy_idx, 1.))
        total.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), .25, error_if_nonfinite=True)
        ref_opt.step()
        with trace.observation(student, torch):
            student(torch.rand(2, 3))
            random.random()
        for actual, expected in zip(student.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert opt.state_dict()['param_groups'] == ref_opt.state_dict()['param_groups']
    for actual, expected in zip(opt.state.values(), ref_opt.state.values()):
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_observation_restores_rng_and_mode_and_rejects_mutation():
    model = torch.nn.Linear(2, 1).train()
    before = torch.get_rng_state().clone()
    py_before = random.getstate()
    with trace.observation(model, torch):
        assert not model.training
        torch.rand(10)
        random.random()
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert py_before == random.getstate()
    with pytest.raises(ValueError, match='modified the model'):
        with trace.observation(model, torch):
            with torch.no_grad():
                model.bias.add_(1)


def test_unverified_window_is_not_reported_as_passing_or_ruled_out():
    steps = [{'step': i, 'short_code_retention': {'passed': i != 3},
              'learning': {'passed': i >= 2}} for i in (1, 2, 3)]
    report = trace.timing_summary(steps, {'1': {'operating_point_found': False},
                                         '3': {'operating_point_found': False}})
    assert report['first_short_code_loss_step'] == 3
    assert report['first_placement_learning_step'] == 2
    assert report['overlap_steps_without_full_evaluation'] == [2]
    assert not report['all_potential_windows_evaluated']
    assert not report['operating_point_found']


def test_learning_gate_requires_both_gains_and_unchanged_denominators():
    cfg = trace.load_config()
    before = {'cases': 24, 'tokens': 170, 'exact_top1': 1, 'token_top1_rate': 88/170}
    assert not trace.placement_learning(before, {**before, 'exact_top1': 2}, cfg)['passed']
    assert not trace.placement_learning(before, {**before, 'token_top1_rate': 94/170}, cfg)['passed']
    assert trace.placement_learning(before, {**before, 'exact_top1': 2, 'token_top1_rate': 94/170}, cfg)['passed']
    with pytest.raises(ValueError, match='denominators'):
        trace.placement_learning(before, {**before, 'tokens': 169}, cfg)


def test_published_config_cannot_silently_drift(tmp_path, monkeypatch):
    config = tmp_path / 'changed.json'
    config.write_bytes(trace.CONFIG.read_bytes().replace(b'1e-07', b'2e-07'))
    monkeypatch.setattr(trace, 'CONFIG', config)
    with pytest.raises(ValueError, match='configuration changed'):
        trace.load_config()


def test_full_evaluation_steps_default_to_the_published_pair():
    cfg = trace.load_config()
    assert trace.full_eval_steps(','.join(str(s) for s in trace.DEFAULT_FULL_EVAL_STEPS), cfg) == {1, 40}


def test_requested_states_are_added_without_dropping_baseline_or_endpoint():
    cfg = trace.load_config()
    assert trace.full_eval_steps('12,13,23', cfg) == {1, 12, 13, 23, 40}
    assert trace.full_eval_steps(' 12 , ,13 ', cfg) == {1, 12, 13, 40}
    assert trace.full_eval_steps('', cfg) == {1, 40}


def test_unmeasurable_states_are_rejected_rather_than_silently_skipped():
    cfg = trace.load_config()
    with pytest.raises(ValueError, match='outside 1..40'):
        trace.full_eval_steps('41', cfg)
    with pytest.raises(ValueError, match='outside 1..40'):
        trace.full_eval_steps('0', cfg)
    with pytest.raises(ValueError, match='positive integers'):
        trace.full_eval_steps('twelve', cfg)
    with pytest.raises(ValueError, match='positive integers'):
        trace.full_eval_steps('-3', cfg)
