import json
from pathlib import Path

import torch

from jobs import ember_first_update_diagnostic as diag


def test_projection_preserves_only_the_unconstrained_direction():
    basis = []
    assert diag.add_basis(torch.tensor([1., 0., 0.]), basis, torch)
    assert diag.add_basis(torch.tensor([1., 1., 0.]), basis, torch)
    assert not diag.add_basis(torch.tensor([2., 2., 0.]), basis, torch)
    delta = torch.tensor([3., -4., 5.])
    actual = diag.project_update(delta, basis, torch)
    torch.testing.assert_close(actual, torch.tensor([0., 0., 5.]))
    assert actual.norm() <= delta.norm()


def test_projection_changes_first_adam_update_and_keeps_empty_source_state():
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.1, .2, .3]]))
    theta = diag.flat_parameters(model, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    assert not optimizer.state
    model(torch.ones(1,3)).sum().backward()
    optimizer.step()
    delta = diag.flat_parameters(model, torch) - theta
    restricted = diag.project_update(delta, [torch.tensor([1.,0.,0.])], torch)
    assert restricted[0] == 0
    assert restricted[1] != 0
    diag.assign_vector(model, theta, torch)
    torch.testing.assert_close(diag.flat_parameters(model, torch), theta, rtol=0, atol=0)


def test_equal_total_score_cannot_hide_a_lost_case():
    def row(i):
        return {"id": i, "score": {"envelope_json_valid": True, "tool_name_correct": True}}
    result = diag.retained_case_ids({"rows": [row("a"), row("b")]}, {"rows": [row("b"), row("c")]})
    assert result == {"passed": False, "source_passed": 2, "retained": 1, "lost_ids": ["a"], "gained_ids": ["c"]}


def test_fresh_targets_are_disjoint_and_match_subtypes():
    used = diag.historical_values()
    original = set(used)
    values = diag.new_target_values(used)
    flat = [v for phase in values.values() for group in phase.values() for v in group]
    assert len(flat) == len(set(flat))
    assert not set(flat) & original
    assert not set(flat) & set(diag.data.copy_data.HELD_OUT_VALUES)
    for groups in values.values():
        for subtype, group in groups.items():
            kind, _ = diag.data.v048d.SUBTYPE_VARIANT[subtype]
            assert all(diag.data.v044.subtype_for(kind, value) == subtype for value in group)


def test_plan_is_fixed_to_one_proposal_two_arms_and_cpu_timeout():
    cfg = json.loads(diag.CONFIG.read_text())
    assert cfg["optimizer_steps"] == 1
    assert cfg["arms"] == ["unrestricted", "projected"]
    assert cfg["learning_rate"] == 1e-7
    assert cfg["wall_time_limit_seconds"] == 1500
    source = Path(diag.__file__).read_text()
    assert source.count("optimizer.step()") == 1
    assert "torch.save(" not in source
